# Verification Submission Guide

This document maps the verification package to the materials requested by the DDL-X Track 3 organizing committee, focusing on rerunning the submitted model pipeline from input images to JSON.

## Single Submitted System

The final leaderboard submission corresponds to one fixed single submitted model solution.
It is packaged as a fixed composite inference system, not as separate alternative submissions.
It is submitted as a single verification package and produces the three required JSON fields together:

```text
Classification result
Bounding boxes
Visible forgery traces
```

The system internally contains three fixed learned modules:

1. a ConvNeXt-B face-level classification branch;
2. a WBF localization branch over `detector_a_fullmask_stageb`, `detector_b_conservative_stageb`, and `detector_c_yolov8m_stageb`;
3. a Qwen2.5-VL-3B-Instruct + LoRA checkpoint-1500 explanation branch conditioned on image, predicted label, and boxes.

These branches are not alternative submissions. They are the components of the same final pipeline used to regenerate a DDL-X Track 3 JSON package from images.

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
src/ddli_classification_v1/ ConvNeXt-B training and test inference code
src/ddli_detector_v1/     detector/WBF training, sweep, merge, and package builders
src/ddli_explain_v1/      Qwen explanation data construction, inference launchers, and scoring
src/ddlx_full_infer_v1/   single public raw-image to JSON inference interface
scripts/                  organizer-facing verification and reproduction entrypoints
configs/                  final WBF and threshold configuration
```

Important entrypoints for model rerun:

```text
scripts/run_end_to_end_from_images.sh
scripts/smoke_end_to_end_from_images.sh
scripts/run_explanation_inference.sh
```

## Environment Files

The package includes:

```text
requirements.txt
environment.yml
```

The original verification environment used CUDA, PyTorch, Ultralytics, ModelScope/Swift, and Qwen-VL utilities. For model reruns of detection and explanation generation, use a GPU environment matching the package dependencies.

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

Provide the official test image directory. The file
stem is used as `image_id`; file stems should be unique.

```text
/path/to/test/images/*.jpg
```

The end-to-end entrypoint creates `test_images.csv`, face/landmark metadata,
classification predictions, detector outputs, WBF boxes, final JSON files, and a
zip package.

## Full Model Rerun

To use the submitted models to regenerate a fresh JSON package from images, run:

```bash
python -m src.ddlx_full_infer_v1.run_end_to_end \
  --image-dir /path/to/test/images \
  --model-root /path/to/model_package/models \
  --out-dir /path/to/output \
  --gpus auto
```

That command reruns the fixed single submitted model solution:

```text
image and face metadata
-> ConvNeXt-B classification branch
-> detector_a_fullmask_stageb + detector_b_conservative_stageb + detector_c_yolov8m_stageb + WBF localization branch
-> fallback boxes when fake and no WBF box is retained
-> prompt construction from image + label + boxes
-> Qwen2.5-VL + LoRA explanation branch
-> final DDL-X JSON zip
```

The regenerated JSON follows the same schema as the submitted Track 3 JSON files. Qwen-generated text may vary slightly across software and CUDA environments.

## Optional Explanation-Only Rerun

```bash
bash scripts/run_explanation_inference.sh \
  /path/to/explain_inputs.jsonl \
  /tmp/ddlx_qwen_verify
```

This invokes:

```text
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

This command is only needed if the verifier wants to rerun the explanation module on prepared prompt JSONL files. The main recommended path is still `src.ddlx_full_infer_v1.run_end_to_end`, which creates the prompt inputs internally.

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

Detailed model and asset checksums are in:

```text
docs/model_manifest.md
```

The full Hugging Face bundle also includes `FILE_MANIFEST.json`.

## Technical Report

The technical report material is provided in:

```text
docs/technical_report.md
docs/training_summary.md
docs/explanation_generation_details.md
docs/compliance_audit.md
```

The workshop paper draft gives the same method-level description in paper form.
