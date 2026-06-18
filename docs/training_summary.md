# Training and Selection Summary

This bundle includes the training launchers and hyperparameters that produced the detector family used by the final WBF submission.

## Explanation model training

The Qwen2.5-VL-3B LoRA adapter can be trained without the original external
reference-code directory:

```bash
cp configs/qwen_lora_sft.env.example configs/qwen_lora_sft.env
bash scripts/train_qwen_lora.sh configs/qwen_lora_sft.env
```

The portable configuration records the checkpoint-1500 run: ms-swift 4.2.1,
eight GPUs, LoRA rank 16, alpha 32, dropout 0.05, all-linear targets, one epoch,
per-device batch size 1, gradient accumulation 4, learning rate 5e-5, cosine
scheduling, warmup ratio 0.03, maximum length 4096, and maximum pixels 602112.
Training and validation JSONL paths must point to authorized challenge data.

## detector_a_fullmask_stageb

Final weight in bundle:

```text
models/detectors/detector_a_fullmask_stageb.pt
```

Legacy internal name: `old`.

Training provenance script:

```text
scripts/training/launch_fullmask_continue96_bestdet_stageb3_6gpu_v3.sh
```

Key settings: YOLO detector, 6 GPUs, image size 512, batch 96, Stage B dev adaptation for 3 epochs, AdamW, lr0 0.0002.

## detector_b_conservative_stageb

Final weight in bundle:

```text
models/detectors/detector_b_conservative_stageb.pt
```

Legacy internal name: `repeat2`.

Training provenance scripts:

```text
scripts/training/launch_fullmask_continue96_currentbest_stageab_repeat_v1.sh
scripts/training/launch_fullmask_continue96_repeat2_conservative_lr1e4_v1.sh
```

Key settings: continued from old/current-best detector, fullmask Stage A with mid-step checkpoints, Stage B dev adaptation for 3 epochs, AdamW, lr0 0.0001 for conservative repeat2.

## detector_c_yolov8m_stageb

Final weight in bundle:

```text
models/detectors/detector_c_yolov8m_stageb.pt
```

Legacy internal name: `yolov8m`.

Training provenance scripts:

```text
scripts/training/launch_hetero_yolov8m512_stagea_v1.sh
scripts/training/watch_then_launch_hetero_yolov8m512_stageb_v1.sh
scripts/training/launch_hetero_yolov8m512_eval_v1.sh
```

Key settings: YOLOv8m COCO initialization, Stage A fullmask training, Stage B dev adaptation for 3 epochs, image size 512, batch 96, 6 GPUs.

## WBF model selection

Final WBF config is frozen in:

```text
configs/final_wbf.yaml
```

The selected WBF setup was `detector_a_fullmask_stageb + detector_b_conservative_stageb + detector_c_yolov8m_stageb`, favoring detector B with weights `0.7 / 1.35 / 1.0` and requiring at least 2 contributing models per kept box.
