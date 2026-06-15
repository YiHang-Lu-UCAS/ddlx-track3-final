# Training and Selection Summary

This bundle includes the training launchers and hyperparameters that produced the detector family used by the final WBF submission.

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
