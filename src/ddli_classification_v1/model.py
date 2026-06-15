from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torchvision.models import convnext_base


def build_convnext_base(
    pretrained_path: str | Path | None = None,
) -> nn.Module:
    model = convnext_base(weights=None)

    if pretrained_path:
        ckpt = torch.load(Path(pretrained_path).expanduser(), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if not isinstance(ckpt, dict):
            raise ValueError("Unsupported pretrained checkpoint format.")

        cleaned: Dict[str, Any] = {}
        for key, value in ckpt.items():
            cleaned[key.removeprefix("module.")] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        classifier_keys = {"classifier.2.weight", "classifier.2.bias"}
        non_classifier_missing = [key for key in missing if key not in classifier_keys]
        non_classifier_unexpected = [key for key in unexpected if key not in classifier_keys]
        if non_classifier_missing or non_classifier_unexpected:
            raise RuntimeError(
                f"Unexpected pretrained mismatch. missing={non_classifier_missing}, "
                f"unexpected={non_classifier_unexpected}"
            )

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)
    return model

