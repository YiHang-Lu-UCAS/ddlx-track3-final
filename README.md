# DDL-X Track 3 Solution

This repository contains the code, configuration, and documentation for our DDL-X Track 3 challenge submission.
The submitted verification package corresponds to one fixed, single submitted model solution used for the final leaderboard submission.
It should be reviewed as a fixed composite inference system rather than as separate alternative submissions.
Internally, the system has a classification branch, a localization branch, and an explanation branch, but these branches are invoked together by the same final pipeline to produce one JSON output per image.

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

- Single submitted system: one fixed inference pipeline that jointly writes the three required DDL-X fields.
- Classification branch: ConvNeXt-B face-level binary classifier.
- Localization branch: WBF over three detector streams, `old`, `repeat2`, and `yolov8m`.
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
text_sources/     README for cached Qwen-generated text artifacts.
evidence/         Lightweight evidence and expected saved-box location.
final_artifact/   Final artifact hash records.
release_assets/   Historical checklist; final large assets are hosted on Hugging Face.
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

Large model weights are not committed directly to git. They are hosted on Hugging Face:

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets
```

To download the model and verification assets into the repository root:

```bash
hf download limitlesstrain/ddlx-track3-final-assets --local-dir .
```

The Hugging Face repository contains the full verification assets, including the Qwen2.5-VL base model copy, LoRA adapter, detector/classifier weights, cached Qwen-generated text sources, final selected zip, and evidence files.

Additional local layout notes are in:

```text
models/README.md
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

For organizer verification, start from the single-system entrypoint:

```bash
bash scripts/run_final_single_pipeline.sh verify
```

The same wrapper also dispatches the exact-artifact rebuild and method-level Qwen explanation rerun modes described below.
See `docs/verification_submission_guide.md` for the complete submission mapping requested by the organizing committee.
If organizers want to rerun the submitted models to regenerate a fresh JSON package, see `docs/full_model_rerun.md`.

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
docs/verification_submission_guide.md
docs/full_model_rerun.md
docs/model_manifest.md
docs/training_summary.md
docs/explanation_generation_details.md
docs/final_submission_evidence.md
configs/final_wbf.yaml
```

Internal development metrics, including dev24k proxy metrics and sampled BERTScore, are documented for model selection. They are not official final leaderboard scores.

## Submission Boundary

This repository is intended for result verification and method-level reproducibility of the same fixed final leaderboard system. It does not claim general state-of-the-art performance and does not report an unsupported final leaderboard score.

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
