# External Assets

Do not commit large model weights or final zip artifacts directly to git.

The final large assets are hosted on Hugging Face:

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets
```

Recommended assets:

1. `convnextb_cls_dev_adapt_head_stage4_last.pt`
2. `detector_a_fullmask_stageb.pt`
3. `detector_b_conservative_stageb.pt`
4. `detector_c_yolov8m_stageb.pt`
5. `models/explanation/qwen2_5_vl_3b_instruct/`
6. `models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/`
7. `text_sources/`
8. `detector_pred_boxes.json`
9. `DDLX_Track3_FinalVerification_WBF_20260530_1431.tar`
10. `submission_fake_nobox_nose_eyes_mouth.zip`

The Qwen2.5-VL-3B-Instruct base-model copy is included in the Hugging Face asset repository for verification, with the official upstream source recorded for provenance.
