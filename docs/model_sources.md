# Model Sources

## Classifier

- Bundle path: `models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt`
- Original path: `/home/pengsiran/projects_data/luyihang/experiments/ddli_cls_dev_adapt_head_stage4_v1/checkpoints/last.pt`
- Model: ConvNeXt-B face-level binary classifier
- Image-level aggregation: `max(face_fake_probability)`
- Final threshold: `fake_prob >= 0.20`

## Detectors

### detector_a_fullmask_stageb

- Bundle path: `models/detectors/detector_a_fullmask_stageb.pt`
- Original path: `/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3/stageb_dev_adapt3/weights/best.pt`
- Legacy internal name: `old`

### detector_b_conservative_stageb

- Bundle path: `models/detectors/detector_b_conservative_stageb.pt`
- Original path: `/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_fullmask_continue96_repeat2_conservative_lr1e4_v1/stageb_dev_adapt3_lr1e4/weights/best.pt`
- Legacy internal name: `repeat2`

### detector_c_yolov8m_stageb

- Bundle path: `models/detectors/detector_c_yolov8m_stageb.pt`
- Original path: `/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1/stageb_dev_adapt3_b96/weights/best.pt`
- Legacy internal name: `yolov8m`

## Explanation Model

### Base VLM

- Bundle path: `models/explanation/qwen2_5_vl_3b_instruct/`
- Original path: `/home/pengsiran/projects_data/luyihang/qwen25vl_sft_repro/models/Qwen2.5-VL-3B-Instruct`
- Size on server: approximately `7.1G`

### LoRA adapter

- Bundle path: `models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/`
- Original path: `/home/pengsiran/projects_data/luyihang/experiments/ddli_explain_qwen25vl3b_lora_v1/formal_run_v1/v0-20260526-222731/checkpoint-1500`
- Key adapter file: `adapter_model.safetensors`
- Checkpoint time: `2026-05-27 00:03:15 +0800`
- Size on server: approximately `344M`

### Explanation generation provenance

The released verification path regenerates explanations from each input image
with this Qwen2.5-VL-3B base model and checkpoint-1500 LoRA adapter. The prompt
also receives the model-predicted label and localized or fallback boxes.
