from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_INDEX = {"real_face": 0, "fake_face": 1}


class DDLIFaceClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.transform = transform
        self.rows: List[Tuple[str, str, Tuple[int, int, int, int] | None, int, int, str | None]] = []

        with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            has_materialized_crop = "crop_image_path" in fieldnames
            for row in reader:
                if row["face_label"] not in LABEL_TO_INDEX:
                    raise ValueError(f"Unexpected face_label={row['face_label']!r}")
                crop_path = row.get("crop_image_path") if has_materialized_crop else None
                crop_bbox = None if crop_path else self._parse_bbox(row["crop_bbox"])
                self.rows.append(
                    (
                        row["image_id"],
                        row["image_label"],
                        crop_bbox,
                        LABEL_TO_INDEX[row["face_label"]],
                        int(row["face_id"]),
                        crop_path,
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, image_id: str, image_label: str) -> Path:
        split_dir = "fake" if image_label == "fake" else "real"
        return self.dataset_root / "train" / split_dir / image_id

    @staticmethod
    def _parse_bbox(raw: str) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = ast.literal_eval(raw)
        return int(x1), int(y1), int(x2), int(y2)

    def __getitem__(self, index: int):
        image_id, image_label, crop_bbox, label, face_id, crop_path = self.rows[index]
        if crop_path:
            image_path = self.dataset_root / crop_path
            with Image.open(image_path) as img:
                crop = img.convert("RGB")
        else:
            image_path = self._image_path(image_id, image_label)
            with Image.open(image_path) as img:
                crop = img.convert("RGB").crop(crop_bbox)

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, label, image_id, face_id


def count_labels(manifest_path: str | Path) -> Dict[str, int]:
    counts = {"real_face": 0, "fake_face": 0}
    with Path(manifest_path).expanduser().resolve().open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["face_label"]] += 1
    return counts
