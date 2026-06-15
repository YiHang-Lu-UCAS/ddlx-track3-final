# Model and Asset Manifest

This manifest lists the model and verification assets for the single integrated DDL-X Track 3 final system.
Large files are hosted at:

```text
https://huggingface.co/limitlesstrain/ddlx-track3-final-assets
```

The full Hugging Face bundle also contains `FILE_MANIFEST.json` with per-file checksums.

## Core Model Files

| Component | Package path | Role in single system | Source / pretrained backbone | Size bytes | SHA256 |
|---|---|---|---|---:|---|
| Classifier branch | `models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt` | Face-level fake probability; image score is max face probability | ConvNeXt-B | 350357358 | `7efbeb496871a524a11dc50fbd8025fda68cd4b333344066559955cf5d3763ea` |
| Localization branch, old | `models/detectors/old_fullmask_continue96_stageb3_best.pt` | YOLO-family detector stream for WBF | Ultralytics YOLO-family detector | 22490915 | `0a5fa22ca148c969d958fb6e7f104a309be4f78f332a4ea5212c30fcbb8ac253` |
| Localization branch, repeat2 | `models/detectors/repeat2_conservative_lr1e4_stageb_best.pt` | YOLO-family detector stream for WBF | Ultralytics YOLO-family detector | 22490915 | `71c300cef6e4162c31be8c58a7f32f7a14ceacc4d5a41efcd09ec45b8407f30e` |
| Localization branch, yolov8m | `models/detectors/yolov8m512_stageab_stageb_best.pt` | YOLOv8m detector stream for WBF | Ultralytics YOLOv8m | 52003147 | `e347e40511a7561d8dba8ef04144bc0327a4ca9781cac24bf93643ed7047ab4c` |
| Explanation branch adapter | `models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/adapter_model.safetensors` | LoRA adapter for visible-forgery explanation generation | Qwen2.5-VL-3B-Instruct + LoRA | 119809088 | `f2d5c9106c43d2a6db29e658e60e207ebde9a083ff0858a9e0c2bdab97c7e8bf` |

## Qwen Base Model Copy

Official upstream source:

```text
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
```

The verification bundle includes the base-model copy used with the LoRA adapter:

| File | Size bytes | SHA256 |
|---|---:|---|
| `models/explanation/qwen2_5_vl_3b_instruct/config.json` | 1373 | `7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5` |
| `models/explanation/qwen2_5_vl_3b_instruct/model-00001-of-00002.safetensors` | 3982649232 | `41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8` |
| `models/explanation/qwen2_5_vl_3b_instruct/model-00002-of-00002.safetensors` | 3526688744 | `365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4` |
| `models/explanation/qwen2_5_vl_3b_instruct/preprocessor_config.json` | 350 | `f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0` |
| `models/explanation/qwen2_5_vl_3b_instruct/tokenizer.json` | 7031645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `models/explanation/qwen2_5_vl_3b_instruct/tokenizer_config.json` | 5702 | `4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56` |

## Verification Assets

| Asset | Purpose | Size bytes | SHA256 |
|---|---|---:|---|
| `evidence/detector_pred_boxes.json` | Saved WBF boxes for exact artifact reconstruction | 5793176 | `d3b09c7ffa4f68055404b3e8f34533399828c6cb2ec7d97a9104dd5072287bea` |
| `text_sources/normal.zip` | Cached Qwen-generated text for normal variant | 137057005 | `943c1738f3538bc808946fe9d04751a09a86db3fec3e4eae7c61f560af984dd3` |
| `text_sources/fake_nobox_eyes.zip` | Cached Qwen-generated text for fake fallback variant | 137292269 | `ea114e9ec04159f00c2db20fccffa66ead6d61fbcdfaed3eb1e40fa6109ea48f` |
| `text_sources/fake_nobox_eyes_mouth.zip` | Cached Qwen-generated text for fake fallback variant | 137391340 | `6a5e76d875d96c9d24f1a4337b8f1ccece077810b0d818569eaa20fc6a8cd51e` |
| `text_sources/fake_nobox_nose_eyes_mouth.zip` | Cached Qwen-generated text for selected fake fallback variant | 137444357 | `1585bdd4890bad8acf71a9efbe76d8c7191f31c53617a88ad281062f39052169` |
| `evidence/final_submission/submission_ddl_x_test_wbf_old_repeat2_yolov8m_pr125_iou035_post175_req2_textreuse_v1_fake_nobox_nose_eyes_mouth.zip` | Final selected leaderboard artifact | 137474463 | `a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97` |

## Fixed Hyperparameters

| Subsystem | Setting |
|---|---|
| Classification threshold | `fake_prob >= 0.20` |
| WBF detector streams | `old`, `repeat2`, `yolov8m` |
| WBF weights | `0.7 / 1.35 / 1.0` |
| WBF pre-confidence | `0.125` |
| WBF IoU | `0.35` |
| WBF post-confidence | `0.175` |
| Maximum boxes per image | `3` |
| Required contributing detector streams | `2` |
| Selected fallback variant | `fake_nobox_nose_eyes_mouth` |
| Explanation adapter | `qwen2_5_vl_3b_lora_checkpoint1500` |
| Method-level Qwen max new tokens | `2048` |

## Notes

- The final zip is exactly reproducible from saved WBF boxes and cached Qwen-generated text sources.
- Method-level explanation reruns use the Qwen base model copy and LoRA adapter listed above.
- Qwen reruns are not expected to reproduce byte-identical text unless the complete software, CUDA, preprocessing, and generation environment is held fixed.
