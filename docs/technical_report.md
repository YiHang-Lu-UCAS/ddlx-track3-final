# Technical Report

## Method summary

This verification package corresponds to one fixed DDL-X Track 3 leaderboard submission. The submitted solution is organized as a single reproducible inference pipeline with three coordinated learned modules and deterministic post-processing. The three learned modules are a ConvNeXt-B face-level classifier, a YOLO-family localization ensemble fused by Weighted Boxes Fusion (WBF), and a Qwen2.5-VL-3B-Instruct explanation model adapted with a LoRA checkpoint. The pipeline produces the three required JSON fields together: `Classification result`, `Bounding boxes`, and `Visible forgery traces`.

The selected fallback behavior is the `fake_nobox_nose_eyes_mouth` variant for fake images without retained WBF boxes.

## Model architecture

The classification module is a ConvNeXt-B face-level binary classifier. Given accepted face crops, it predicts a fake-face probability for each face. The image-level score is computed as:

```text
image_fake_prob = max(face_fake_probability)
```

The final image label uses threshold:

```text
fake_prob >= 0.20
```

The localization module uses three fixed detector streams:

```text
detector_a_fullmask_stageb      -> models/detectors/detector_a_fullmask_stageb.pt
detector_b_conservative_stageb  -> models/detectors/detector_b_conservative_stageb.pt
detector_c_yolov8m_stageb       -> models/detectors/detector_c_yolov8m_stageb.pt
```

Their detections are merged by WBF with the final frozen configuration:

```text
weights = 0.7 / 1.35 / 1.0
pre_conf = 0.125
wbf_iou = 0.35
post_conf = 0.175
max_boxes = 3
require_models = 2
```

The explanation module uses:

```text
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

It receives the image, predicted label, and predicted or fallback boxes through a generated prompt, then produces the `Visible forgery traces` text.

## Training strategy

The classifier was trained as a face-level fake/real classifier using ConvNeXt-B. The training code is under `src/ddli_classification_v1/`, with the final bundle weight at:

```text
models/classifier/convnextb_cls_dev_adapt_head_stage4_last.pt
```

The detector family was trained and selected through the scripts preserved under `scripts/training/` and `src/ddli_detector_v1/`. Detector A uses a fullmask continuation followed by Stage B development adaptation. Detector B continues from the earlier detector family with a conservative learning rate. Detector C starts from a YOLOv8m COCO initialization before Stage A fullmask training and Stage B development adaptation. The recorded detector settings include image size 512, batch size 96, six GPUs, AdamW optimization, and three epochs of Stage B adaptation.

The explanation model uses Qwen2.5-VL-3B-Instruct as the base model and a LoRA adapter at checkpoint-1500. The adapter was selected using an internal sampled BERTScore evaluation, documented in `docs/explanation_generation_details.md` and the evidence files under `evidence/explanation/`. This BERTScore evaluation is not an official challenge metric.

## Data usage

The package records the use of DDL-I and DDL-X challenge data paths in the training and provenance scripts. The classifier consumes face-crop manifests. The detector training uses fullmask and development-adaptation stages. The explanation model is trained and evaluated from challenge-derived image, label, box, and explanation inputs.

For verification reruns, the required external input is the official DDL-X test image directory. The end-to-end inference entrypoint creates image manifests, face and landmark metadata, classifier predictions, detector outputs, WBF boxes, final JSON files, and a submission zip. No manual labels, manual boxes, or manually written explanations are used in the final output generation pipeline.

## Post-processing

Post-processing is deterministic. The classifier output is aggregated from face-level probabilities to an image-level score with a fixed threshold. Detector outputs are merged by WBF with the fixed parameters listed above. If an image is predicted as fake and no WBF box is retained, the selected final variant generates fallback boxes around the nose, left eye, right eye, and mouth from model-derived face landmarks and deterministic geometry.

The final JSON assembly follows the official Track 3 schema:

```json
{
  "Bounding boxes": [[x1, y1, x2, y2]],
  "Visible forgery traces": "model-generated explanation text",
  "Classification result": "fake"
}
```

For real images, `Bounding boxes` is `null`. Bounding boxes are integer coordinates on the DDL-X 1-1000 scale.

## Explanation generation pipeline

The explanation pipeline constructs Qwen-VL prompts from the input image, predicted classification label, and predicted or fallback boxes. For fake images, the prompt asks the model to describe visually supportable manipulation traces and to end with a concise tampering statement. For real images, the prompt asks for visually supportable evidence of consistency and a non-tampering statement. The model is instructed not to output JSON or coordinates; only the explanation field is replaced by generated text.

The method-level rerun command uses Swift inference:

```bash
swift infer \
  --model models/explanation/qwen2_5_vl_3b_instruct \
  --adapters models/explanation/qwen2_5_vl_3b_lora_checkpoint1500 \
  --template qwen2_5_vl \
  --infer_backend pt \
  --torch_dtype float16 \
  --max_length 4096 \
  --max_pixels 602112 \
  --max_new_tokens 2048 \
  --temperature 0 \
  --num_beams 1
```

Model reruns may produce semantically similar but byte-different Qwen text if software, CUDA kernels, or preprocessing differ.
To avoid incomplete explanation fields, the released end-to-end merge step uses a completeness safeguard: empty Qwen responses fall back to the deterministic base statement, and non-empty responses that lack the expected final conclusion receive an appended summary statement.

## Reproduce JSON from images

The verification package regenerates a fresh JSON package from input images:

```bash
python -m src.ddlx_full_infer_v1.run_end_to_end \
  --image-dir /path/to/test/images \
  --model-root /path/to/model_package/models \
  --out-dir /path/to/output \
  --gpus auto
```

This path checks that the submitted model package can produce classification, localization, and explanation outputs from images.

## Asset and checksum references

The complete model inventory, pretrained backbone sources, file sizes, and checksums are recorded in:

```text
docs/model_manifest.md
docs/model_sources.md
FILE_MANIFEST.json
```

The main pretrained backbones are ConvNeXt-B, Ultralytics YOLO-family detectors including YOLOv8m, and Qwen2.5-VL-3B-Instruct. The upstream Qwen model source is:

```text
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
```

## Boundary

This report describes the final challenge submission and its model-rerun verification package. It does not claim general state-of-the-art performance, does not report an unsupported final leaderboard score, and does not claim that Qwen reruns will reproduce byte-identical text. Model-level reproducibility is provided by the released weights, prompts, and inference scripts.

The released package is intended for model-level reproduction from input images to JSON. Since the explanation field is regenerated by Qwen2.5-VL with the released LoRA adapter, minor wording differences may occur compared with the original leaderboard submission, especially under different software, CUDA, or inference-library environments. In addition, a very small number of bounding boxes or serialization details may differ because of preprocessing, detector-library versions, JSON formatting, or zip metadata. Therefore, the regenerated zip is not expected to be byte-identical to the leaderboard submission; the verification target is the released model workflow, weights, configuration, and output schema.
