# Full Model Rerun to Regenerate JSON

This document describes the flow for organizers who want to use the submitted model package to regenerate a fresh DDL-X Track 3 JSON submission.

This is different from exact artifact reconstruction:

- exact artifact reconstruction uses saved WBF boxes and cached Qwen-generated text to reproduce the final leaderboard zip/hash;
- full model rerun invokes the submitted model branches again to produce a new JSON package from images and metadata.

The rerun is expected to produce valid DDL-X JSON files with the same fields and method, but Qwen-generated text may not be byte-identical to the cached final artifact unless the full software, CUDA, preprocessing, and generation environment is held fixed.

## Inputs

Required inputs:

```text
DDL-X image directory
test_images.csv with image_id, image_path, width, height
face preprocessing outputs with accepted face crops and landmarks
models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
models/detectors/old_fullmask_continue96_stageb3_best.pt
models/detectors/repeat2_conservative_lr1e4_stageb_best.pt
models/detectors/yolov8m512_stageab_stageb_best.pt
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

The original server launcher and paths are preserved in:

```text
scripts/launch_wbf_hetero_fulltest_4variants_v1.sh
```

In a clean organizer environment, replace those absolute paths with local paths to the DDL-X image directory, metadata, and downloaded model weights.

## Output

The final rerun output should be a zip containing one JSON file per test image:

```text
json/<image_id>.json
```

Each JSON file uses the official Track 3 field names:

```json
{
  "Bounding boxes": [[x1, y1, x2, y2]],
  "Visible forgery traces": "model-generated explanation text",
  "Classification result": "fake"
}
```

For real images, `Bounding boxes` is `null`.
Box coordinates are integer values on the DDL-X 1-1000 scale.

## End-to-End Flow

### 1. Prepare image and face metadata

Create or provide:

```text
test_images.csv
seg_face_shards/shard_*.csv
raw_face_outputs/shard_*/faces.jsonl
```

The face metadata must include the accepted face rows and model-derived landmarks used by the deterministic fallback box generator.

### 2. Run the classification branch

Run the ConvNeXt-B face-level classifier on accepted face crops:

```text
models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
```

Expected intermediate outputs:

```text
cls_preds/shard_*.csv
image_scores.csv
```

The image-level score is:

```text
image_fake_prob = max(face_fake_probability)
```

The final label threshold is:

```text
fake_prob >= 0.20
```

### 3. Run the localization branch

Run the three detector streams over the face/image metadata:

```text
old      -> models/detectors/old_fullmask_continue96_stageb3_best.pt
repeat2  -> models/detectors/repeat2_conservative_lr1e4_stageb_best.pt
yolov8m  -> models/detectors/yolov8m512_stageab_stageb_best.pt
```

Expected detector outputs:

```text
old_det_shards/shard_*.csv
repeat2_det_shards/shard_*.csv
yolov8m_det_shards/shard_*.csv
```

Merge detections with:

```text
src/ddli_detector_v1/merge_wbf_test_boxes_hetero_v1.py
```

WBF settings:

```text
weights = old:0.7, repeat2:1.35, yolov8m:1.0
pre_conf = 0.125
wbf_iou = 0.35
post_conf = 0.175
max_boxes = 3
require_models = 2
```

The WBF output is:

```text
detector_pred_boxes.json
```

### 4. Build base JSON with labels and boxes

Combine:

```text
image_scores.csv
test_images.csv
cls_preds/shard_*.csv
raw_face_outputs/shard_*/faces.jsonl
detector_pred_boxes.json
```

For fake images:

- use WBF boxes when retained;
- if no WBF box is retained, use deterministic nose/eyes/mouth fallback boxes from model-derived landmarks.

For real images:

- set `Bounding boxes` to `null`.

At this stage, the base JSON directory contains the model-produced classification and localization fields.
The explanation field may be a placeholder until the Qwen branch is run.

### 5. Construct Qwen prompt inputs

Build Qwen/VL JSONL shards from the base JSON and image paths:

```bash
python src/ddli_explain_v1/build_test_explain_shards.py \
  --submission-json-dir /path/to/base_json \
  --test-images /path/to/test_images.csv \
  --out-dir /path/to/qwen_inputs \
  --num-shards 6
```

This script constructs prompts from:

```text
input image
predicted Classification result
predicted or fallback Bounding boxes
```

### 6. Run the Qwen explanation branch

Run Qwen2.5-VL-3B-Instruct with the LoRA checkpoint-1500 adapter:

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model models/explanation/qwen2_5_vl_3b_instruct \
  --adapters models/explanation/qwen2_5_vl_3b_lora_checkpoint1500 \
  --template qwen2_5_vl \
  --val_dataset /path/to/qwen_inputs/shard_00.jsonl \
  --result_path /path/to/qwen_predictions/shard_00_predictions.jsonl \
  --max_new_tokens 2048 \
  --max_length 4096 \
  --max_pixels 602112 \
  --temperature 0 \
  --num_beams 1
```

Repeat for all shards, or adapt the preserved six-GPU launcher:

```text
src/ddli_explain_v1/launch_test_explain_full_6gpu_v1.sh
```

### 7. Merge explanations into final JSON and zip

Merge Qwen responses back into the base JSON:

```bash
python src/ddli_explain_v1/merge_test_explanations_submission.py \
  --source-json-dir /path/to/base_json \
  --manifest-dir /path/to/qwen_inputs \
  --prediction-dir /path/to/qwen_predictions \
  --output-json-dir /path/to/final_json \
  --zip-path /path/to/submission_model_rerun.zip \
  --summary-path /path/to/submission_model_rerun_summary.json \
  --num-shards 6
```

The resulting zip is a fresh model-generated DDL-X Track 3 submission package.

### 8. Validate output

Validate that:

```text
there are 100000 JSON files;
all files contain Classification result, Bounding boxes, and Visible forgery traces;
real images have Bounding boxes = null;
Visible forgery traces is non-empty;
the final zip can be hashed.
```

Relevant validation utilities:

```text
src/ddli_cls_v1/validate_ddlx_submission.py
src/ddli_explain_v1/validate_multibox_text_submission.py
```

## Relationship to the Final Leaderboard Artifact

The exact final leaderboard artifact is:

```text
evidence/final_submission/submission_ddl_x_test_wbf_old_repeat2_yolov8m_pr125_iou035_post175_req2_textreuse_v1_fake_nobox_nose_eyes_mouth.zip
```

with SHA256:

```text
a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97
```

If organizers rerun the full model pipeline, the regenerated Qwen explanations may be textually different from the cached final explanations.
This is why the package provides two verification views:

1. exact artifact reconstruction for leaderboard hash verification;
2. full/model-level rerun for checking that the submitted model solution can generate all three required fields.
