# Training and Selection Summary

This bundle includes the training launchers and hyperparameters that produced the detector family used by the final WBF submission.

## old detector

Final weight in bundle:

```text
models/detectors/old_fullmask_continue96_stageb3_best.pt
```

Training provenance script:

```text
scripts/training/launch_fullmask_continue96_bestdet_stageb3_6gpu_v3.sh
```

Key settings: YOLO detector, 6 GPUs, image size 512, batch 96, Stage B dev adaptation for 3 epochs, AdamW, lr0 0.0002.

## repeat2 detector

Final weight in bundle:

```text
models/detectors/repeat2_conservative_lr1e4_stageb_best.pt
```

Training provenance scripts:

```text
scripts/training/launch_fullmask_continue96_currentbest_stageab_repeat_v1.sh
scripts/training/launch_fullmask_continue96_repeat2_conservative_lr1e4_v1.sh
```

Key settings: continued from old/current-best detector, fullmask Stage A with mid-step checkpoints, Stage B dev adaptation for 3 epochs, AdamW, lr0 0.0001 for conservative repeat2.

## YOLOv8m detector

Final weight in bundle:

```text
models/detectors/yolov8m512_stageab_stageb_best.pt
```

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

The selected WBF setup was `old + repeat2 + yolov8m`, favoring repeat2 with weights `0.7 / 1.35 / 1.0` and requiring at least 2 contributing models per kept box.
