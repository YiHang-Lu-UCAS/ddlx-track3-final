from __future__ import annotations
import argparse,csv,gzip,json,pickle,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, '/home/pengsiran/projects_data/luyihang/ddli_segmentation_v1')
from eval_bboxes_against_json import load_manifest_unique,load_json_boxes
from eval_region_iou_multibox import union_area

def overlap(a,b):
 ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); inter=ix*iy; aa=(a[2]-a[0])*(a[3]-a[1]); bb=(b[2]-b[0])*(b[3]-b[1]); u=aa+bb-inter
 return (inter/u if u else 0.,inter/min(aa,bb) if min(aa,bb)>0 else 0.)
def config(path): return json.load(open(path))['selected']
def face_scores(path):
 d=defaultdict(list)
 with open(path,newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f): d[r['image_id']].append(float(r['fake_prob']))
 return {k:max(v) for k,v in d.items()}
def build_predictions(recs,score,cfg,cthr):
 raw=defaultdict(list); gate={i for i,p in score.items() if p>=cthr}; t=str(float(cfg['mask_threshold']))
 for r in recs:
  if r['image_id'] not in gate: continue
  for x in r['components'][t]:
   b=x['bbox1000']; ar=(b[2]-b[0])*(b[3]-b[1]); side=min(b[2]-b[0],b[3]-b[1])
   if ar>=cfg['min_box_area_1000'] and side>=cfg['min_box_side_1000']: raw[r['image_id']].append({'bbox':x['bbox'],'score':x['score'],'area':ar})
 out={}
 for iid,items in raw.items():
  keep=[]
  for x in sorted(items,key=lambda q:(q['score'],q['area']),reverse=True):
   if any((lambda z:z[0]>cfg['nms_iou'] or z[1]>cfg['containment_threshold'])(overlap(x['bbox'],k['bbox'])) for k in keep): continue
   keep.append(x)
   if len(keep)>=int(cfg['max_boxes_per_image']): break
  if keep: out[iid]={'Bounding boxes':[x['bbox'] for x in keep]}
 return out,gate
def evaluate(pred,gate,info):
 fake_iou=fake_n=fake_zero=real_false=boxes=inter_sum=union_sum=0; tp=tn=fp=fn=0
 for iid,r in info.items():
  truth=r['image_label']=='fake'; gp=load_json_boxes(r['json_path'],image_path=r['image_path'],reference_size=1024); pp=pred.get(iid,{}).get('Bounding boxes',[]); pred_fake=iid in gate
  if truth and pred_fake:tp+=1
  elif truth:fn+=1
  elif pred_fake:fp+=1
  else:tn+=1
  ga=union_area(gp);pa=union_area(pp);ua=union_area(gp+pp);inter=ga+pa-ua; v=inter/ua if ua else 1.
  if truth: fake_n+=1;fake_iou+=v;fake_zero+=v==0
  elif pp:real_false+=1
  boxes+=len(pp);inter_sum+=inter;union_sum+=ua
 real_n=sum(r['image_label']=='real' for r in info.values())
 return {'classification_acc':(tp+tn)/len(info),'classification_tp_fake':tp,'classification_fn_fake':fn,'classification_fp_real':fp,'classification_tn_real':tn,'fake_image_region_iou_mean':fake_iou/fake_n,'micro_area_region_iou_over_all_images':inter_sum/union_sum,'fake_zero_iou_rate':fake_zero/fake_n,'real_false_box_rate':real_false/real_n,'num_pred_boxes':boxes,'pred_fake_images':len(gate)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--face-predictions',required=True);ap.add_argument('--image-manifest',required=True);ap.add_argument('--cleanup-config',required=True);ap.add_argument('--thresholds',default='0.20,0.25,0.30,0.35,0.40,0.45,0.48,0.50,0.55,0.60,0.65');ap.add_argument('--lock-threshold',type=float,default=None);ap.add_argument('--out-json',required=True);ap.add_argument('--out-pred-json',required=True);a=ap.parse_args()
 with gzip.open(a.cache,'rb') as f: recs=pickle.load(f)['records']
 info=load_manifest_unique(Path(a.image_manifest)); scores=face_scores(a.face_predictions); cfg=config(a.cleanup_config); thrs=[a.lock_threshold] if a.lock_threshold is not None else [float(x) for x in a.thresholds.split(',')]; results=[]; preds={}
 for t in thrs:
  p,g=build_predictions(recs,scores,cfg,t); m=evaluate(p,g,info);m['classification_threshold']=t; results.append(m);preds[str(t)]=p
 results.sort(key=lambda x:(x['fake_image_region_iou_mean'],-x['real_false_box_rate'],x['classification_acc']),reverse=True); best=results[0]['fake_image_region_iou_mean']; elig=[x for x in results if best-x['fake_image_region_iou_mean']<.002];elig.sort(key=lambda x:(x['real_false_box_rate'],-x['fake_image_region_iou_mean'],-x['classification_acc']));sel=elig[0]
 out={'fixed_localization_config':cfg,'ranking_rule':'maximize fake region IoU; within 0.002 lower real false-box rate','selected':sel,'results_ranked':results};Path(a.out_json).parent.mkdir(parents=True,exist_ok=True);Path(a.out_json).write_text(json.dumps(out,indent=2),encoding='utf-8');Path(a.out_pred_json).write_text(json.dumps(preds[str(sel['classification_threshold'])],indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__': main()
