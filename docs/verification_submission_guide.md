# Verification Submission Guide

This document maps the verification package to the materials requested by the DDL-X Track 3 organizing committee.

## Single Submitted System

The final leaderboard submission corresponds to one fixed single submitted model solution.
It is packaged as a fixed composite inference system, not as separate alternative submissions.
It is submitted as a single verification package and produces the three required JSON fields together:

```text
Classification result
Bounding boxes
Visible forgery traces
```

The system internally contains three fixed branches:

1. a ConvNeXt-B face-level classification branch;
2. a WBF localization branch over the `old`, `repeat2`, and `yolov8m` detector streams;
3. a Qwen2.5-VL-3B-Instruct + LoRA checkpoint-1500 explanation branch conditioned on image, predicted label, and boxes.

These branches are not alternative submissions. They are the components of the same final pipeline used to generate the selected leaderboard artifact.

## Links

Code repository:

```text
https://github.com/YiHang-Lu-UCAS/ddlx-track3-final
```

Model and verification assets:

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets
```

Official upstream Qwen base model:

```text
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
```

## Source Code

Training, inference, evaluation, and post-processing code are organized as follows:

```text
src/ddli_cls_v1/          classification utilities and submission validation
src/ddli_detector_v1/     detector/WBF training, sweep, merge, and package builders
src/ddli_explain_v1/      Qwen explanation data construction, inference launchers, and scoring
scripts/                  organizer-facing verification and reproduction entrypoints
configs/                  final WBF and threshold configuration
```

Important entrypoints:

```text
scripts/run_final_single_pipeline.sh
scripts/verify_bundle.sh
scripts/rebuild_from_saved_wbf_boxes.sh
scripts/run_explanation_inference.sh
scripts/launch_wbf_hetero_fulltest_4variants_v1.sh
```

## Environment Files

The package includes:

```text
requirements.txt
environment.yml
```

The original verification environment used CUDA, PyTorch, Ultralytics, ModelScope/Swift, and Qwen-VL utilities.
For exact artifact verification, only standard shell utilities and `sha256sum` are required.
For method-level reruns of detection or explanation generation, use a GPU environment matching the package dependencies.

## Model Files

Large model files are hosted on Hugging Face rather than committed directly to git.
Download the assets with:

```bash
hf download limitlesstrain/ddlx-track3-final-assets --local-dir hf_assets
```

The Hugging Face repository stores the complete verification bundle under:

```text
DDLX_Track3_FinalVerification_WBF_20260530_1431/
```

To run commands from this repository layout, copy or symlink the contents of that bundle into the repository root, or set paths explicitly when invoking scripts.
The model inventory and checksums are listed in `docs/model_manifest.md`.

## Data Preparation

For exact hash verification of the selected final zip, no dataset images are required.

For exact artifact reconstruction from saved WBF boxes, provide the original DDL-X metadata and face preprocessing outputs:

```text
image_scores.csv
test_images.csv
cls_preds/
raw face metadata/output directory
```

The original server paths are preserved in `scripts/launch_wbf_hetero_fulltest_4variants_v1.sh` as provenance.
In a clean verification environment, replace those paths with the corresponding local DDL-X metadata paths.

For method-level Qwen explanation verification, prepare a JSONL file in the same format used by `src/ddli_explain_v1/`:

```text
image path
predicted label
predicted or fallback boxes
prompt content for Visible forgery traces
```

## Full Model Rerun

If the organizing committee wants to use the submitted models to regenerate a fresh JSON package, use:

```text
docs/full_model_rerun.md
```

That flow reruns the fixed single submitted model solution:

```text
image and face metadata
-> ConvNeXt-B classification branch
-> three detector streams + WBF localization branch
-> fallback boxes when fake and no WBF box is retained
-> prompt construction from image + label + boxes
-> Qwen2.5-VL + LoRA explanation branch
-> final DDL-X JSON zip
```

The regenerated JSON follows the same schema as the final leaderboard artifact.
It may not be byte-identical because Qwen text generation can differ across software and CUDA environments.

## Inference and Verification Commands

### 1. Verify the exact submitted artifact

```bash
bash scripts/run_final_single_pipeline.sh verify
```

or:

```bash
bash scripts/verify_bundle.sh final_artifact/submission_fake_nobox_nose_eyes_mouth.zip
```

Expected SHA256:

```text
a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97
```

### 2. Rebuild from saved WBF boxes and cached generated text

```bash
bash scripts/run_final_single_pipeline.sh rebuild \
  /path/to/image_scores.csv \
  /path/to/test_images.csv \
  /path/to/cls_preds \
  /path/to/raw_face_outputs \
  /tmp/ddlx_rebuild_wbf
```

This path uses:

```text
evidence/detector_pred_boxes.json
text_sources/normal.zip
text_sources/fake_nobox_eyes.zip
text_sources/fake_nobox_eyes_mouth.zip
text_sources/fake_nobox_nose_eyes_mouth.zip
```

It is the exact artifact reconstruction path for the final leaderboard package.

### 3. Rerun the explanation branch at method level

```bash
bash scripts/run_final_single_pipeline.sh qwen \
  /path/to/explain_inputs.jsonl \
  /tmp/ddlx_qwen_verify
```

This invokes:

```text
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

The regenerated text may be semantically similar but byte-different from the cached text used in the selected final zip.
Use the cached `text_sources/` zips when checking exact hash equivalence.

## Input and Output Formats

Each output JSON follows the DDL-X Track 3 field names:

```json
{
  "Bounding boxes": [[x1, y1, x2, y2]],
  "Visible forgery traces": "model-generated explanation text",
  "Classification result": "fake"
}
```

For real images:

```json
{
  "Bounding boxes": null,
  "Visible forgery traces": "model-generated explanation text",
  "Classification result": "real"
}
```

Bounding boxes are integer coordinates on the DDL-X 1-1000 scale.

## Checksums

The final selected zip checksum is:

```text
a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97
```

Detailed model and asset checksums are in:

```text
docs/model_manifest.md
```

The full Hugging Face bundle also includes `FILE_MANIFEST.json`.

## Technical Report

The technical report material is provided in:

```text
docs/training_summary.md
docs/explanation_generation_details.md
docs/compliance_audit.md
docs/final_submission_evidence.md
```

The workshop paper draft gives the same method-level description in paper form.
