#!/usr/bin/env python3
"""Validate DDL-X Track 3 JSON submission files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_KEYS = {"Bounding boxes", "Visible forgery traces", "Classification result"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def validate_one(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path.name}: invalid json: {exc}"]
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        errors.append(f"{path.name}: missing keys {sorted(missing)}")
    label = payload.get("Classification result")
    bbox = payload.get("Bounding boxes")
    text = payload.get("Visible forgery traces")
    if label not in {"real", "fake"}:
        errors.append(f"{path.name}: bad label {label!r}")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{path.name}: empty explanation")
    if label == "real" and bbox is not None:
        errors.append(f"{path.name}: real sample must have null bbox")
    if label == "fake":
        if not (isinstance(bbox, list) and len(bbox) == 4):
            errors.append(f"{path.name}: fake sample must have 4-number bbox")
        else:
            for value in bbox:
                if not isinstance(value, int) or not (1 <= value <= 1000):
                    errors.append(f"{path.name}: bbox value out of range/int: {bbox}")
                    break
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                errors.append(f"{path.name}: bbox order invalid: {bbox}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--json-dir", required=True)
    args = parser.parse_args()

    images = list_images(Path(args.image_dir))
    json_dir = Path(args.json_dir)
    image_stems = {p.stem for p in images}
    json_paths = sorted(json_dir.rglob("*.json"))
    json_stems = {p.stem for p in json_paths}

    errors: list[str] = []
    missing = sorted(image_stems - json_stems)
    extra = sorted(json_stems - image_stems)
    if missing:
        errors.append(f"missing_json={len(missing)} first={missing[:5]}")
    if extra:
        errors.append(f"extra_json={len(extra)} first={extra[:5]}")

    labels = {"real": 0, "fake": 0}
    for path in json_paths:
        errors.extend(validate_one(path))
        try:
            labels[json.loads(path.read_text(encoding="utf-8")).get("Classification result", "")] += 1
        except Exception:
            pass

    print(f"images={len(images)} json={len(json_paths)} real={labels['real']} fake={labels['fake']}")
    if errors:
        print("VALIDATION FAILED")
        for err in errors[:50]:
            print(err)
        raise SystemExit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
