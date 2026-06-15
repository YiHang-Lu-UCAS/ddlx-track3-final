from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from model import build_convnext_base


class FakeProbabilityModel(nn.Module):
    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(images))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the DDL-I ConvNeXt-B face classifier to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--opset", default=17, type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint.expanduser(), map_location="cpu")
    classifier = build_convnext_base(None)
    classifier.load_state_dict(checkpoint["model"], strict=True)
    model = FakeProbabilityModel(classifier).eval()
    sample = torch.randn(1, 3, args.input_size, args.input_size)

    with torch.no_grad():
        torch_output = model(sample)
    torch.onnx.export(
        model,
        sample,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["fake_probability"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "fake_probability": {0: "batch_size"},
        },
    )
    metadata = {
        "model": "DDL-I ConvNeXt-B face-level binary classifier",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "onnx": str(args.output),
        "onnx_sha256": sha256(args.output),
        "opset": args.opset,
        "input": {
            "name": "images",
            "shape": ["batch_size", 3, args.input_size, args.input_size],
            "dtype": "float32",
            "color": "RGB",
            "preprocess": {
                "resize": [args.input_size, args.input_size],
                "scale": "uint8 RGB to float32 in [0, 1]",
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        },
        "output": {
            "name": "fake_probability",
            "shape": ["batch_size", 1],
            "dtype": "float32",
            "meaning": "sigmoid probability that the input face crop is fake",
        },
        "image_level_aggregation_outside_onnx": "max(face_fake_probability)",
        "main_pipeline_image_threshold_outside_onnx": 0.20,
        "sample_output": torch_output.tolist(),
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
