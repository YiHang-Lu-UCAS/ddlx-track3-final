# DDL-X Track 3 Solution

This repository contains the code, configuration, and documentation for our DDL-X Track 3 challenge submission.
The submitted verification package corresponds to one fixed, single submitted model solution used for the final leaderboard submission.
It should be reviewed as a fixed composite inference system rather than as separate alternative submissions.
Internally, the system has a classification branch, a localization branch, and an explanation branch, but these branches are invoked together by the same final pipeline to produce one JSON output per image.

The system produces three required outputs for each test image:

1. image-level fake/real classification;
2. localization boxes for visible manipulated regions;
3. text explanations for visible forgery traces.

The model-rerun entrypoint regenerates a fresh JSON package from input images.
The selected fallback behavior for fake images without retained WBF boxes is:

```text
fake_nobox_nose_eyes_mouth
```

## Method Summary

- Single submitted system: one fixed inference pipeline that jointly writes the three required DDL-X fields.
- Classification branch: ConvNeXt-B face-level binary classifier.
- Localization branch: WBF over three detector streams, `detector_a_fullmask_stageb`, `detector_b_conservative_stageb`, and `detector_c_yolov8m_stageb`.
- Fallback post-processing: for fake images without retained WBF boxes, boxes are generated from model-derived face landmarks and deterministic geometry.
- Explanation branch: Qwen2.5-VL-3B-Instruct plus LoRA checkpoint-1500, conditioned on the image, predicted label, and boxes.

All output fields are produced by trained models and deterministic post-processing. The explanations are not manually written, and the fallback boxes are not manually drawn.

## Repository Contents

```text
configs/          Final WBF and system configuration.
scripts/          Top-level verification and reproduction entrypoints.
src/              Classifier, detector, WBF, and explanation scripts.
docs/             Reproduction notes, model sources, training summary, and compliance audit.
models/           README and expected model-weight layout.
```

## Setup

The verified vision and Qwen stacks used different PyTorch versions. Create
both locked environments:

```bash
conda env create -f environment.yml
conda activate ddlx-track3-solution
conda env create -f environment-qwen.yml
```

Equivalent pip lock files are:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-qwen-lock.txt
```

`environment.yml` runs preprocessing, classification, localization, WBF, and
JSON assembly. `environment-qwen.yml` runs Qwen LoRA training and inference.
Pass the Qwen environment's Swift executable to the public entrypoint with
`--swift-command /path/to/ddlx-track3-qwen/bin/swift`.

The recorded vision stack uses PyTorch 2.3.1+cu118, torchvision 0.18.1+cu118,
Ultralytics 8.4.53, and timm 0.6.11. The Qwen stack uses PyTorch 2.6.0+cu118,
ms-swift 4.2.1, PEFT 0.19.1, and qwen-vl-utils 0.0.14.

## Model Weights

Large model weights are not committed directly to git. They are hosted on Hugging Face:

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets
```

To download the model-rerun bundle:

```bash
hf download limitlesstrain/ddlx-track3-final-assets \
  --include "DDLX_Track3_ModelRerunOnly_20260616/**" \
  --local-dir hf_assets
```

The self-contained bundle root is
`hf_assets/DDLX_Track3_ModelRerunOnly_20260616/`. Run commands from that
directory, where `models/`, `src/`, `scripts/`, and the environment files are
siblings. Alternatively, copy its contents into a GitHub checkout.

The Hugging Face repository contains the model-rerun assets, including the Qwen2.5-VL base model copy, LoRA adapter, detector/classifier weights, source code, scripts, configuration files, and documentation.

Additional local layout notes are in:

```text
models/README.md
```

Expected paths after downloading assets:

```text
models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
models/detectors/detector_a_fullmask_stageb.pt
models/detectors/detector_b_conservative_stageb.pt
models/detectors/detector_c_yolov8m_stageb.pt
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

## Reproduce JSON From Images

Use the single public inference entrypoint:

```bash
python -m src.ddlx_full_infer_v1.run_end_to_end \
  --image-dir /path/to/test/images \
  --model-root /path/to/model_package/models \
  --out-dir /path/to/output \
  --gpus auto
```

The equivalent bash wrapper is `scripts/run_end_to_end_from_images.sh`.
If Swift is installed in a separate environment, add `--swift-command /path/to/env/bin/swift`.

## Explanation Training

The Qwen LoRA training entrypoint is self-contained:

```bash
cp configs/qwen_lora_sft.env.example configs/qwen_lora_sft.env
# Edit the four input/output paths in configs/qwen_lora_sft.env.
bash scripts/train_qwen_lora.sh configs/qwen_lora_sft.env
```

Challenge training data are not redistributed. Authorized users must prepare
the ms-swift JSONL records with the released data-construction scripts.

This command scans input images, runs preprocessing, classification, localization,
fallback box generation, Qwen explanation generation, validates the JSON/zip
schema and file counts, and writes:

```text
/path/to/output/final_json/*.json
/path/to/output/submission_model_rerun.zip
/path/to/output/run_summary.json
```

See `docs/reproduce_from_images.md` for the complete raw-image reproduction flow.

## Training and Selection Notes

See:

```text
docs/verification_submission_guide.md
docs/reproduce_from_images.md
docs/full_model_rerun.md
docs/technical_report.md
docs/model_manifest.md
docs/training_summary.md
docs/explanation_generation_details.md
configs/final_wbf.yaml
```

Internal development metrics, including dev24k proxy metrics and sampled BERTScore, are documented for model selection. They are not official final leaderboard scores.

## Submission Boundary

This repository is intended for result verification by rerunning the submitted model pipeline from input images to JSON. It does not claim general state-of-the-art performance and does not report an unsupported final leaderboard score.

## External Asset Link

Model and verification assets:

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets
```

Code repository:

```text
https://github.com/YiHang-Lu-UCAS/ddlx-track3-final
```

## Citation

The workshop paper should cite the required dataset papers:

- MFFI: Multi-Dimensional Face Forgery Image Dataset for Real-World Scenarios.
- DDL: A Large-Scale Datasets for Deepfake Detection and Localization in Diversified Real-World Scenarios.

See `docs/citation_notes.md`.
