# Release Assets

Do not commit large model weights or final zip artifacts directly to git.

Attach these files to a GitHub Release such as `v1.0-final`, or provide the same files through an organizer-approved storage link.

Recommended assets:

1. `convnextb_cls_dev_adapt_head_stage4_last.pt`
2. `old_fullmask_continue96_stageb3_best.pt`
3. `repeat2_conservative_lr1e4_stageb_best.pt`
4. `yolov8m512_stageab_stageb_best.pt`
5. `qwen2_5_vl_3b_lora_checkpoint1500.tar.gz`
6. `text_sources.tar.gz`
7. `detector_pred_boxes.json`
8. `DDLX_Track3_FinalVerification_WBF_20260530_1431.tar`
9. `submission_fake_nobox_nose_eyes_mouth.zip`

The Qwen2.5-VL-3B-Instruct base model is approximately 7.1 GB. Prefer an official model-source download or a separate model-hosting link if it is too large for GitHub Release.
