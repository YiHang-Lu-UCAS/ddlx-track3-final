# DDL-I ConvNeXt-B classification v1

This package trains the first-stage face-level classifier:

```text
face crop RGB -> ConvNeXt-B -> fake_face / real_face
```

It is designed to consume the finished `metadata_v1` manifests directly:

```text
metadata_v1/train_faces.csv
metadata_v1/val_faces.csv
```

## Files

```text
ddli_classification_v1/
|-- data.py
|-- model.py
|-- train.py
|-- eval.py
|-- launch_train_8gpu.sh
`-- README.md
```

## Runtime assumptions

- Use the existing server environment:

```bash
/home/pengsiran/anaconda3/envs/vmamba/bin/python
```

- The server is treated as offline for pretrained weights.
- A local pretrained ConvNeXt-B checkpoint must be supplied through:

```bash
PRETRAINED_PATH=/absolute/path/to/convnext_base-6075fbad.pth
```

## Training command

```bash
cd ~/projects_data/luyihang

PRETRAINED_PATH=/absolute/path/to/convnext_base-6075fbad.pth \
bash ddli_classification_v1/launch_train_8gpu.sh
```

Default output directory:

```text
~/projects_data/luyihang/experiments/ddli_cls_convnextb_v1
```

## Outputs

```text
run_config.json
metrics.csv
checkpoints/
  |-- last.pt
  |-- best_auc.pt
  `-- best_acc.pt
```

## Evaluation

```bash
/home/pengsiran/anaconda3/envs/vmamba/bin/python ddli_classification_v1/eval.py \
  --dataset-root ~/projects_data/luyihang/datasets/DDL-I \
  --manifest ~/projects_data/luyihang/datasets/DDL-I/metadata_v1/val_faces.csv \
  --checkpoint ~/projects_data/luyihang/experiments/ddli_cls_convnextb_v1/checkpoints/best_auc.pt \
  --amp
```

## Notes on the first training version

- Crop extraction is online from the original images using `crop_bbox`.
- Labels are:
  - `real_face -> 0`
  - `fake_face -> 1`
- `pos_weight` is computed from the accepted training manifest at runtime.
- The default augmentations are intentionally mild:
  - fixed resize to `224 x 224`
  - horizontal flip
  - light color jitter
- Checkpoints are saved by:
  - best AUC
  - best accuracy
- This package trains the classifier only. It is intentionally separated from the later FPN segmentation stage.
