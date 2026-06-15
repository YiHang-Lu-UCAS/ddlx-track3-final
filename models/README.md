# Model Weights

This repository tracks code, configs, and documentation. Large model weights should be downloaded from the Hugging Face asset repository or the organizer verification bundle.

## Required Weights

### Classifier

- Expected path: `models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt`
- Model: ConvNeXt-B face-level binary classifier
- Image-level aggregation: `max(face_fake_probability)`
- Final fake threshold: `0.20`

### Detectors

- Expected path: `models/detectors/detector_a_fullmask_stageb.pt`
- Expected path: `models/detectors/detector_b_conservative_stageb.pt`
- Expected path: `models/detectors/detector_c_yolov8m_stageb.pt`

Legacy asset names are still accepted by the inference code for backward
compatibility:

- `old_fullmask_continue96_stageb3_best.pt`
- `repeat2_conservative_lr1e4_stageb_best.pt`
- `yolov8m512_stageab_stageb_best.pt`

These three detector streams are fused with WBF using `configs/final_wbf.yaml`.

### Explanation Model

- Base VLM expected path: `models/explanation/qwen2_5_vl_3b_instruct/`
- LoRA adapter expected path: `models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/`
- Adapter checkpoint: `checkpoint-1500`
- Key adapter file: `adapter_model.safetensors`

The final selected WBF submission reused cached Qwen-generated text artifacts for exact zip reconstruction. Method-level verification should rerun the Qwen base model plus the LoRA adapter on a small subset or full metadata.

## Hugging Face Asset Layout

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets

DDLX_Track3_FinalVerification_WBF_20260530_1431/
  models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
  models/detectors/detector_a_fullmask_stageb.pt
  models/detectors/detector_b_conservative_stageb.pt
  models/detectors/detector_c_yolov8m_stageb.pt
  models/explanation/qwen2_5_vl_3b_instruct/
  models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
  text_sources/
  detector_pred_boxes.json
  DDLX_Track3_FinalVerification_WBF_20260530_1431.tar
```

The Hugging Face asset repository includes the Qwen2.5-VL-3B-Instruct base-model copy used for verification. The upstream model page is also recorded for provenance.
