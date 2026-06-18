#!/usr/bin/env python3
"""Validate DDL-X Track 3 JSON submission files."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_KEYS = {"Bounding boxes", "Visible forgery traces", "Classification result"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def validate_one(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
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
        boxes = [bbox] if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox) else bbox
        if not isinstance(boxes, list) or not boxes:
            errors.append(f"{path.name}: fake sample must have at least one bbox")
        else:
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    errors.append(f"{path.name}: bbox must contain four coordinates: {box!r}")
                    continue
                for value in box:
                    if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 1000):
                        errors.append(f"{path.name}: bbox value out of range/int: {box}")
                        break
                if all(isinstance(value, int) and not isinstance(value, bool) for value in box):
                    if box[0] >= box[2] or box[1] >= box[3]:
                        errors.append(f"{path.name}: bbox order invalid: {box}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--zip-path")
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
            labels[json.loads(path.read_text(encoding="utf-8-sig")).get("Classification result", "")] += 1
        except Exception:
            pass

    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.is_file():
            errors.append(f"missing_zip={zip_path}")
        else:
            with zipfile.ZipFile(zip_path) as archive:
                bad_entry = archive.testzip()
                names = archive.namelist()
                expected_names = {f"json/{stem}.json" for stem in image_stems}
                actual_names = set(names)
                if bad_entry is not None:
                    errors.append(f"corrupt_zip_entry={bad_entry}")
                if len(names) != len(actual_names):
                    errors.append("zip contains duplicate entry names")
                missing_zip = expected_names - actual_names
                extra_zip = actual_names - expected_names
                if missing_zip:
                    errors.append(f"missing_zip_json={len(missing_zip)} first={sorted(missing_zip)[:5]}")
                if extra_zip:
                    errors.append(f"extra_zip_entries={len(extra_zip)} first={sorted(extra_zip)[:5]}")

    print(f"images={len(images)} json={len(json_paths)} real={labels['real']} fake={labels['fake']}")
    if errors:
        print("VALIDATION FAILED")
        for err in errors[:50]:
            print(err)
        raise SystemExit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
