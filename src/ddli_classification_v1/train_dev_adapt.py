from __future__ import annotations
import argparse,csv,math,os,time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np,torch,torch.distributed as dist
from sklearn.metrics import accuracy_score,average_precision_score,precision_recall_fscore_support,roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data import count_labels, LABEL_TO_INDEX
from PIL import Image
import ast
from model import build_convnext_base
from utils import append_csv_row,is_main_process,save_json,seed_everything,strip_module_prefix


class DevFaceAdaptDataset(Dataset):
 def __init__(self, manifest, transform=None):
  self.transform=transform; self.rows=[]
  with open(manifest,newline='',encoding='utf-8') as f:
   for r in csv.DictReader(f):
    b=ast.literal_eval(r['crop_bbox']); self.rows.append((r['image_path'], tuple(int(round(float(x))) for x in b), LABEL_TO_INDEX[r['face_label']], r['image_id'], int(r['face_id'])))
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  path,b,y,iid,fid=self.rows[i]
  with Image.open(path) as im: crop=im.convert('RGB').crop(b)
  if self.transform: crop=self.transform(crop)
  return crop,y,iid,fid

def args():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--train-manifest',required=True);p.add_argument('--val-manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--init-checkpoint',required=True);p.add_argument('--input-size',type=int,default=224);p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--num-workers',type=int,default=6);p.add_argument('--head-lr',type=float,default=1e-4);p.add_argument('--stage-lr',type=float,default=1e-5);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--seed',type=int,default=20260525);p.add_argument('--amp',action='store_true');p.add_argument('--log-interval',type=int,default=50);return p.parse_args()
def setup():
 if 'RANK' not in os.environ:return 0,1,0
 r=int(os.environ['RANK']);ws=int(os.environ['WORLD_SIZE']);lr=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(lr);dist.init_process_group('nccl');return r,ws,lr
def metric(y,p,t):
 z=(p>=t).astype(np.int64);pr,re,f1,_=precision_recall_fscore_support(y,z,average='binary',zero_division=0);return {'acc':float(accuracy_score(y,z)),'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'precision':float(pr),'recall':float(re),'f1':float(f1)}
def gather(x,ws):
 if ws==1:return x
 t=torch.as_tensor(x,device='cuda'); sizes=[torch.zeros(1,device='cuda',dtype=torch.long) for _ in range(ws)]; n=torch.tensor([t.shape[0]],device='cuda');dist.all_gather(sizes,n);mx=max(int(s.item()) for s in sizes)
 if t.shape[0]<mx:t=torch.cat([t,torch.zeros((mx-t.shape[0],)+t.shape[1:],device='cuda',dtype=t.dtype)])
 arr=[torch.zeros_like(t) for _ in range(ws)];dist.all_gather(arr,t);return np.concatenate([a[:int(s.item())].cpu().numpy() for a,s in zip(arr,sizes)])
def main():
 a=args();rank,ws,local=setup();seed_everything(a.seed+rank);device=torch.device('cuda',local);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 tr=transforms.Compose([transforms.Resize((a.input_size,a.input_size)),transforms.RandomHorizontalFlip(.5),transforms.ColorJitter(.1,.1,.05,.02),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); va=transforms.Compose([transforms.Resize((a.input_size,a.input_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 td=DevFaceAdaptDataset(a.train_manifest,tr); vd=DevFaceAdaptDataset(a.val_manifest,va); ts=DistributedSampler(td,num_replicas=ws,rank=rank,shuffle=True,seed=a.seed); vs=DistributedSampler(vd,num_replicas=ws,rank=rank,shuffle=False)
 tl=DataLoader(td,batch_size=a.batch_size,sampler=ts,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0);vl=DataLoader(vd,batch_size=a.batch_size,sampler=vs,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
 model=build_convnext_base(None); ck=torch.load(a.init_checkpoint,map_location='cpu'); model.load_state_dict({str(k).removeprefix('module.'):v for k,v in ck['model'].items()},strict=True)
 for p in model.parameters():p.requires_grad=False
 for p in model.classifier.parameters():p.requires_grad=True
 for p in model.features[7].parameters():p.requires_grad=True
 trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]; head=[p for n,p in trainable if n.startswith('classifier.')];stage=[p for n,p in trainable if n.startswith('features.7.')]
 model=model.to(device); model=DDP(model,device_ids=[local],output_device=local) if ws>1 else model; counts=count_labels(a.train_manifest);pw=torch.tensor([counts['real_face']/max(1,counts['fake_face'])],device=device);criterion=nn.BCEWithLogitsLoss(pos_weight=pw);opt=AdamW([{'params':head,'lr':a.head_lr},{'params':stage,'lr':a.stage_lr}],weight_decay=a.weight_decay);sc=torch.cuda.amp.GradScaler(enabled=a.amp)
 if is_main_process():save_json(out/'run_config.json',{'args':vars(a),'world_size':ws,'train_size':len(td),'val_size':len(vd),'train_label_counts':counts,'trainable_parameters':sum(p.numel() for _,p in trainable),'trainable_names':[n for n,_ in trainable],'head_lr':a.head_lr,'stage_lr':a.stage_lr})
 fields=['epoch','train_loss','val_loss','acc','auc','ap','precision','recall','f1','head_lr','stage_lr','epoch_seconds'];raw=model.module if isinstance(model,DDP) else model
 for ep in range(a.epochs):
  begin=time.time();ts.set_epoch(ep);model.train();total=seen=0
  for st,(im,y,_,_) in enumerate(tl,1):
   im=im.to(device,non_blocking=True);y=y.float().to(device,non_blocking=True).unsqueeze(1);opt.zero_grad(set_to_none=True)
   with torch.cuda.amp.autocast(enabled=a.amp):loss=criterion(model(im),y)
   sc.scale(loss).backward();sc.step(opt);sc.update(); total+=loss.item()*im.size(0);seen+=im.size(0)
   if is_main_process() and st%a.log_interval==0:print(f'epoch={ep} step={st}/{len(tl)} train_loss={total/seen:.5f}',flush=True)
  model.eval();vloss=vseen=0;pp=[];yy=[]
  with torch.no_grad():
   for im,y,_,_ in vl:
    im=im.to(device,non_blocking=True); y=y.float().to(device,non_blocking=True).unsqueeze(1)
    with torch.cuda.amp.autocast(enabled=a.amp): logits=model(im); loss=criterion(logits,y)
    vloss+=loss.item()*im.size(0);vseen+=im.size(0);pp.append(torch.sigmoid(logits).squeeze(1).cpu().numpy());yy.append(y.squeeze(1).cpu().numpy())
  pn=gather(np.concatenate(pp),ws);yn=gather(np.concatenate(yy),ws).astype(np.int64)
  if is_main_process():
   m=metric(yn,pn,a.threshold);row={'epoch':ep,'train_loss':total/seen,'val_loss':vloss/vseen,**m,'head_lr':a.head_lr,'stage_lr':a.stage_lr,'epoch_seconds':time.time()-begin};append_csv_row(out/'metrics.csv',row,fields); torch.save({'epoch':ep,'model':strip_module_prefix(raw.state_dict()),'args':vars(a),'source_checkpoint':a.init_checkpoint,'metrics':m},out/'checkpoints/last.pt'); print(row,flush=True)
  if ws>1:dist.barrier()
 if ws>1:dist.destroy_process_group()
if __name__=='__main__':main()
