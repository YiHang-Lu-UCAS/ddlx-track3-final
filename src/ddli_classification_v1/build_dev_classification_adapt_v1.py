from __future__ import annotations
import csv,json,random
from pathlib import Path
from collections import Counter,defaultdict
META=Path('/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I/dev/metadata_v1')
OUT=META/'classification_adapt_v1'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260525; rng=random.Random(SEED)
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
 with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
allrows=read(META/'dev_faces.csv'); fields=list(allrows[0]); calib={r['image_id'] for r in read(META/'dev_faces_localization_calib12k_seed20260524.csv')}; hold={r['image_id'] for r in read(META/'dev_faces_localization_holdout12k_seed20260524.csv')}
by=defaultdict(list)
for r in allrows: by[r['image_id']].append(r)
avail={i:rs for i,rs in by.items() if i not in calib and i not in hold}
# Reserve an internal monitoring val set by image; it is not used for final selection.
cat=defaultdict(list)
for iid,rs in avail.items():
 has_pos=any(r['face_label']=='fake_face' for r in rs); image_lab=rs[0]['image_label']
 k='fake_pos' if has_pos else ('fake_no_pos' if image_lab=='fake' else 'real')
 cat[k].append(iid)
val_ids=set()
for k,n in [('fake_pos',2000),('fake_no_pos',1000),('real',3000)]:
 ids=cat[k][:]; rng.shuffle(ids); val_ids.update(ids[:min(n,len(ids))])
train_pool=[r for iid,rs in avail.items() if iid not in val_ids for r in rs]
val=[r for iid,rs in avail.items() if iid in val_ids for r in rs]
pos=[r for r in train_pool if r['face_label']=='fake_face']
fake_neg=[r for r in train_pool if r['face_label']=='real_face' and r['image_label']=='fake']
real_neg=[r for r in train_pool if r['face_label']=='real_face' and r['image_label']=='real']
def sample(pool,n,group):
 out=[]
 for j in range(n):
  r=dict(rng.choice(pool)); r['adapt_group']=group; r['adapt_sample_index']=str(j); out.append(r)
 return out
fields2=fields+['adapt_group','adapt_sample_index']
train=sample(pos,40000,'fake_face_positive')+sample(fake_neg,25000,'fake_image_hard_negative')+sample(real_neg,35000,'real_negative')
rng.shuffle(train)
# attach bookkeeping fields to val without resampling
val2=[]
for j,r in enumerate(val):
 x=dict(r); x['adapt_group']='internal_val'; x['adapt_sample_index']=str(j); val2.append(x)
write(OUT/'train_manifest.csv',train,fields2);write(OUT/'internal_val_manifest.csv',val2,fields2)
summary={'seed':SEED,'all_dev_images':len(by),'excluded_final_calib_images':len(calib),'excluded_final_holdout_images':len(hold),'excluded_intersection':len(calib&hold),'available_images':len(avail),'internal_val_images':len(val_ids),'train_unique_images':len({r['image_id'] for r in train}),'train_rows':len(train),'internal_val_rows':len(val2),'train_components':{'fake_face_positive':40000,'fake_image_hard_negative':25000,'real_negative':35000},'source_pool_counts':{'positive':len(pos),'fake_hard_negative':len(fake_neg),'real_negative':len(real_neg)},'internal_val_face_labels':dict(Counter(r['face_label'] for r in val2)),'leak_checks':{'train_vs_calib_images':len({r['image_id'] for r in train}&calib),'train_vs_holdout_images':len({r['image_id'] for r in train}&hold),'val_vs_calib_images':len(val_ids&calib),'val_vs_holdout_images':len(val_ids&hold),'train_vs_internal_val_images':len({r['image_id'] for r in train}&val_ids)}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
