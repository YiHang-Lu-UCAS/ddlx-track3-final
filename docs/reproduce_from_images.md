# Reproduce JSON from Input Images

This document describes the raw-image reproduction path for the DDL-X Track 3
verification package.

The submitted solution is packaged as one callable model package with one public
inference entrypoint. Internally, the package contains multiple learned modules:

- ConvNeXt-B classification module;
- YOLO/WBF localization module;
- Qwen2.5-VL-3B + LoRA explanation module.

These modules are invoked jointly by the same pipeline and produce one final JSON
file per input image.

## Inputs

Required:

```text
/path/to/test/images/
/path/to/model_package/models/
```

The image directory may contain `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`
files. The file stem is used as `image_id`, so stems must be unique.

The model root must contain:

```text
models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
models/detectors/detector_a_fullmask_stageb.pt
models/detectors/detector_b_conservative_stageb.pt
models/detectors/detector_c_yolov8m_stageb.pt
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

## Command

```bash
python -m src.ddlx_full_infer_v1.run_end_to_end \
  --image-dir /path/to/test/images \
  --model-root /path/to/model_package/models \
  --out-dir /path/to/output \
  --gpus auto
```

The equivalent bash wrapper is:

```bash
bash scripts/run_end_to_end_from_images.sh \
  --image-dir /path/to/test/images \
  --model-root /path/to/model_package/models \
  --out-dir /path/to/output \
  --gpus auto
```

If `swift` is installed in a separate environment, pass its executable path:

```bash
python -m src.ddlx_full_infer_v1.run_end_to_end \
  --image-dir /path/to/test/images \
  --model-root /path/to/model_package/models \
  --out-dir /path/to/output \
  --gpus auto \
  --swift-command /path/to/env/bin/swift
```

GPU selection is configurable:

```text
--gpus auto        respect the verifier's current CUDA_VISIBLE_DEVICES setting
--gpus cpu         CPU smoke test path
--gpus 0           single GPU
--gpus 0,1,2,3     selected GPUs
```

## Pipeline

```text
input images
-> test image manifest
-> MTCNN face/landmark preprocessing
-> ConvNeXt-B face classification
-> image score = max(face fake probability), threshold 0.20
-> detector_a_fullmask_stageb / detector_b_conservative_stageb / detector_c_yolov8m_stageb detector streams
-> weighted box fusion
-> fallback nose/eyes/mouth boxes when fake and no WBF box is retained
-> base JSON
-> Qwen2.5-VL + LoRA explanation generation
-> final JSON directory and zip
```

## Outputs

```text
/path/to/output/final_json/*.json
/path/to/output/submission_model_rerun.zip
/path/to/output/run_summary.json
/path/to/output/pipeline_summary.json
```

Each JSON file has the official Track 3 fields:

```json
{
  "Bounding boxes": [[x1, y1, x2, y2]],
  "Visible forgery traces": "model-generated explanation text",
  "Classification result": "fake"
}
```

For real images, `Bounding boxes` is `null`.

## Smoke Test

For a quick code-path check on a small image folder:

```bash
python -m src.ddlx_full_infer_v1.run_end_to_end \
  --image-dir /path/to/small_image_dir \
  --model-root /path/to/model_package/models \
  --out-dir /tmp/ddlx_smoke \
  --gpus auto \
  --skip-qwen \
  --force
```

The smoke mode uses `--skip-qwen`, so it validates image scanning,
preprocessing, classification, localization, WBF, fallback, JSON writing, and zip
creation without running the slower Qwen explanation model.

For a short Qwen smoke test, reduce `--qwen-max-new-tokens` to a small value
such as `128`. This validates Qwen loading and merge format, but the generated
text may be truncated before the expected final statement. The default full
rerun setting is `2048`.

## Exact Artifact vs Model Rerun

This raw-image path regenerates a fresh JSON package from the submitted model
package. Qwen-generated text may not be byte-identical to the historical
leaderboard artifact unless the software, CUDA, preprocessing, and generation
environment are identical.

For exact byte-level verification of the highest-scoring submitted zip, use the
separate SHA256 and saved-artifact reconstruction procedures documented in
`docs/reproduce_inference.md`.
