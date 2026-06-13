# Model Weights

This repository tracks code, configs, and documentation. Large model weights should be downloaded from the GitHub Release assets, Hugging Face, or the organizer verification bundle.

## Required Weights

### Classifier

- Expected path: `models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt`
- Model: ConvNeXt-B face-level binary classifier
- Image-level aggregation: `max(face_fake_probability)`
- Final fake threshold: `0.20`

### Detectors

- Expected path: `models/detectors/old_fullmask_continue96_stageb3_best.pt`
- Expected path: `models/detectors/repeat2_conservative_lr1e4_stageb_best.pt`
- Expected path: `models/detectors/yolov8m512_stageab_stageb_best.pt`

These three detector streams are fused with WBF using `configs/final_wbf.yaml`.

### Explanation Model

- Base VLM expected path: `models/explanation/qwen2_5_vl_3b_instruct/`
- LoRA adapter expected path: `models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/`
- Adapter checkpoint: `checkpoint-1500`
- Key adapter file: `adapter_model.safetensors`

The final selected WBF submission reused cached Qwen-generated text artifacts for exact zip reconstruction. Method-level verification should rerun the Qwen base model plus the LoRA adapter on a small subset or full metadata.

## Recommended Release Layout

```text
v1.0-final/
  convnextb_cls_dev_adapt_head_stage4_last.pt
  old_fullmask_continue96_stageb3_best.pt
  repeat2_conservative_lr1e4_stageb_best.pt
  yolov8m512_stageab_stageb_best.pt
  qwen2_5_vl_3b_lora_checkpoint1500.tar.gz
  text_sources.tar.gz
  detector_pred_boxes.json
  DDLX_Track3_FinalVerification_WBF_20260530_1431.tar
```

If the Qwen2.5-VL-3B-Instruct base model is not attached to the Release, download it from the official model source and place it at the expected path above.
