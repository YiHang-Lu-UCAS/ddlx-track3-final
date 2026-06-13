# DDL-X Track 3 Solution

This repository contains the code, configuration, and documentation for our DDL-X Track 3 challenge submission.

The system produces three required outputs for each test image:

1. image-level fake/real classification;
2. localization boxes for visible manipulated regions;
3. text explanations for visible forgery traces.

The final selected variant is:

```text
fake_nobox_nose_eyes_mouth
```

Final selected zip SHA256:

```text
a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97
```

## Method Summary

- Classification: ConvNeXt-B face-level binary classifier.
- Localization: WBF over three detector streams, `old`, `repeat2`, and `yolov8m`.
- Fallback boxes: for fake images without retained WBF boxes, boxes are generated from model-derived face landmarks and deterministic geometry.
- Explanations: Qwen2.5-VL-3B-Instruct plus LoRA checkpoint-1500.

All output fields are produced by trained models and deterministic post-processing. The explanations are not manually written, and the fallback boxes are not manually drawn.

## Repository Contents

```text
configs/          Final WBF and system configuration.
scripts/          Top-level verification and reproduction entrypoints.
src/              Classifier, detector, WBF, and explanation scripts.
docs/             Reproduction notes, model sources, training summary, and compliance audit.
models/           README and expected model-weight layout.
text_sources/     README for cached Qwen-generated text artifacts.
evidence/         Lightweight evidence and expected saved-box location.
final_artifact/   Final artifact hash records.
release_assets/   Checklist for files to attach to GitHub Release.
```

## Setup

Create an environment with either conda:

```bash
conda env create -f environment.yml
conda activate ddlx-track3-solution
```

or pip:

```bash
python -m pip install -r requirements.txt
```

CUDA, PyTorch, Ultralytics, ModelScope/Swift, and Qwen-VL utilities should match the verification machine.

## Model Weights

Large model weights are not committed directly to git. See:

```text
models/README.md
release_assets/README.md
```

Expected paths after downloading assets:

```text
models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
models/detectors/old_fullmask_continue96_stageb3_best.pt
models/detectors/repeat2_conservative_lr1e4_stageb_best.pt
models/detectors/yolov8m512_stageab_stageb_best.pt
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

## Reproducibility Modes

### Level 1: exact final artifact verification

This checks that the selected final zip matches the recorded SHA256:

```bash
bash scripts/verify_bundle.sh
```

### Level 2: rebuild from saved WBF boxes

This path rebuilds the four WBF variants using:

- saved WBF boxes;
- cached Qwen-generated text zips under `text_sources/`;
- `src/ddli_detector_v1/build_wbf_text_from_variant_zips.py`.

```bash
bash scripts/rebuild_from_saved_wbf_boxes.sh \
  /path/to/image_scores.json \
  /path/to/test_images.json \
  /path/to/cls_preds_dir \
  /path/to/raw_face_dir \
  /tmp/ddlx_rebuild_wbf
```

### Level 3: method-level Qwen explanation reproduction

This path reruns explanation generation from the model stack:

- Qwen2.5-VL-3B-Instruct;
- LoRA checkpoint-1500;
- image paths, predicted labels, and boxes.

```bash
bash scripts/run_explanation_inference.sh /path/to/explain_inputs.jsonl /tmp/ddlx_qwen_verify
```

The regenerated text may be semantically similar but byte-different from cached final text. The cached `text_sources/` files are used only for exact leaderboard artifact reconstruction.

## Training and Selection Notes

See:

```text
docs/training_summary.md
docs/explanation_generation_details.md
docs/final_submission_evidence.md
configs/final_wbf.yaml
```

Internal development metrics, including dev24k proxy metrics and sampled BERTScore, are documented for model selection. They are not official final leaderboard scores.

## Submission Boundary

This repository is intended for result verification and method-level reproducibility. It does not claim general state-of-the-art performance and does not report an unsupported final leaderboard score.

## Citation

The workshop paper should cite the required dataset papers:

- MFFI: Multi-Dimensional Face Forgery Image Dataset for Real-World Scenarios.
- DDL: A Large-Scale Datasets for Deepfake Detection and Localization in Diversified Real-World Scenarios.

See `docs/citation_notes.md`.
