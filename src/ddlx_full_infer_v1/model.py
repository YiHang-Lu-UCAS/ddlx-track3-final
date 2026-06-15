from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DDLXTrack3Model:
    """Single public model package interface for DDL-X Track 3 inference.

    The package has one callable entrypoint. Internally it loads task-specific
    learned modules for classification, localization, and explanation.
    """

    model_root: Path
    config_path: Path | None = None

    @property
    def classifier_checkpoint(self) -> Path:
        return self.model_root / "classifier" / "convnextb_cls_dev_adapt_head_stage4_last.pt"

    @property
    def detector_checkpoints(self) -> dict[str, Path]:
        return {
            "detector_a_fullmask_stageb": self._first_existing(
                self.model_root / "detectors" / "detector_a_fullmask_stageb.pt",
                self.model_root / "detectors" / "old_fullmask_continue96_stageb3_best.pt",
            ),
            "detector_b_conservative_stageb": self._first_existing(
                self.model_root / "detectors" / "detector_b_conservative_stageb.pt",
                self.model_root / "detectors" / "repeat2_conservative_lr1e4_stageb_best.pt",
            ),
            "detector_c_yolov8m_stageb": self._first_existing(
                self.model_root / "detectors" / "detector_c_yolov8m_stageb.pt",
                self.model_root / "detectors" / "yolov8m512_stageab_stageb_best.pt",
            ),
        }

    @staticmethod
    def _first_existing(preferred: Path, fallback: Path) -> Path:
        return preferred if preferred.exists() else fallback

    @property
    def qwen_base(self) -> Path:
        return self.model_root / "explanation" / "qwen2_5_vl_3b_instruct"

    @property
    def qwen_lora(self) -> Path:
        return self.model_root / "explanation" / "qwen2_5_vl_3b_lora_checkpoint1500"

    def validate_files(self, require_qwen: bool = True) -> None:
        required = [self.classifier_checkpoint, *self.detector_checkpoints.values()]
        if require_qwen:
            required.extend([self.qwen_base, self.qwen_lora])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing model package files:\n" + "\n".join(missing))
