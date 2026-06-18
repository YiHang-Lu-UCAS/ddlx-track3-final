from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
try:
    from .eval_bboxes_against_json import load_json_boxes, parse_box_list, load_manifest_unique
except ImportError:  # pragma: no cover - supports direct script execution
    from eval_bboxes_against_json import load_json_boxes, parse_box_list, load_manifest_unique

def clean_rects(boxes):
    out=[]
    for b in boxes:
        if len(b) < 4: continue
        x1,y1,x2,y2=map(float,b[:4])
        if x2>x1 and y2>y1: out.append([x1,y1,x2,y2])
    return out

def union_area(rects):
    rects=clean_rects(rects)
    if not rects: return 0.0
    xs=sorted({r[0] for r in rects} | {r[2] for r in rects})
    total=0.0
    for xa, xb in zip(xs, xs[1:]):
        if xb <= xa: continue
        intervals=[]
        for x1,y1,x2,y2 in rects:
            if x1 < xb and x2 > xa:
                intervals.append((y1,y2))
        if not intervals: continue
        intervals.sort()
        cur_a,cur_b=intervals[0]; ysum=0.0
        for a,b in intervals[1:]:
            if a > cur_b:
                ysum += cur_b-cur_a; cur_a,cur_b=a,b
            elif b > cur_b:
                cur_b=b
        ysum += cur_b-cur_a
        total += (xb-xa)*ysum
    return total

def main():
    ap=argparse.ArgumentParser(description='Compute per-image union-region IoU for multi-box predictions.')
    ap.add_argument('--manifest', required=True); ap.add_argument('--pred-json', required=True)
    ap.add_argument('--out-json', required=True); ap.add_argument('--out-csv', required=True)
    ap.add_argument('--gt-coordinate-size', type=int, default=1024)
    args=ap.parse_args()
    manifest=load_manifest_unique(Path(args.manifest).resolve())
    preds=json.loads(Path(args.pred_json).read_text(encoding='utf-8'))
    rows=[]; sums={'fake':0.0,'real':0.0,'all_empty_as_one':0.0,'active':0.0}; counts={'fake':0,'real':0,'active':0,'fake_zero_iou':0,'fake_pred_empty':0,'real_false_box':0,'empty_empty':0}
    gt_area_total=pred_area_total=combined_area_total=0.0
    for image_id in sorted(manifest):
        info=manifest[image_id]
        gt=clean_rects(load_json_boxes(info.get('json_path',''), image_path=info.get('image_path',''), reference_size=args.gt_coordinate_size))
        item=preds.get(image_id,{})
        pred=clean_rects(parse_box_list(item.get('Bounding boxes') if isinstance(item,dict) else item))
        ga=union_area(gt); pa=union_area(pred); ua=union_area(gt+pred); ia=max(0.0,ga+pa-ua)
        has_union=ua>0
        iou=(ia/ua) if has_union else 1.0
        label='fake' if gt else 'real'
        counts[label]+=1; sums[label]+=iou; sums['all_empty_as_one']+=iou
        if has_union:
            counts['active']+=1; sums['active']+=iou
            gt_area_total+=ga; pred_area_total+=pa; combined_area_total+=ua
        else: counts['empty_empty']+=1
        if gt and iou == 0: counts['fake_zero_iou']+=1
        if gt and not pred: counts['fake_pred_empty']+=1
        if not gt and pred: counts['real_false_box']+=1
        rows.append({'image_id':image_id,'label':label,'num_gt':len(gt),'num_pred':len(pred),'gt_union_area':ga,'pred_union_area':pa,'intersection_area':ia,'union_area':ua,'region_iou':iou})
    global_area_iou=(gt_area_total+pred_area_total-combined_area_total)/combined_area_total if combined_area_total else 1.0
    def mean(k): return sums[k]/counts[k] if counts[k] else 0.0
    summary={
      'metric_definition':'For each image, union all GT rectangles and all predicted rectangles, then IoU = intersection area / union area. Overlapping boxes are counted once.',
      'num_images':len(manifest),'num_prediction_images':len(preds),
      'fake_images':counts['fake'],'real_images':counts['real'],'active_union_images':counts['active'],'empty_empty_images':counts['empty_empty'],
      'fake_image_region_iou_mean':mean('fake'),
      'active_image_region_iou_mean':mean('active'),
      'all_image_region_iou_mean_empty_empty_as_one':sums['all_empty_as_one']/len(manifest) if manifest else 0.0,
      'micro_area_region_iou_over_all_images':global_area_iou,
      'fake_zero_iou_rate':counts['fake_zero_iou']/counts['fake'] if counts['fake'] else 0.0,
      'fake_no_pred_box_rate':counts['fake_pred_empty']/counts['fake'] if counts['fake'] else 0.0,
      'real_false_box_rate':counts['real_false_box']/counts['real'] if counts['real'] else 0.0,
      'area_totals':{'gt_union_area_sum':gt_area_total,'pred_union_area_sum':pred_area_total,'intersection_area_sum':gt_area_total+pred_area_total-combined_area_total,'union_area_sum':combined_area_total},
      'pred_json':str(Path(args.pred_json).resolve())
    }
    Path(args.out_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with Path(args.out_csv).open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
