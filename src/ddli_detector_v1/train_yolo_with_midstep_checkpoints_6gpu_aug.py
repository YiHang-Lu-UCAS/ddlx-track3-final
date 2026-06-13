from __future__ import annotations

import argparse
import json
import shutil

from ultralytics import YOLO
from ultralytics.utils import LOGGER, RANK


def parse_steps(raw: str) -> set[int]:
    return {int(value) for value in raw.split(",") if value.strip()}


def parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {raw}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint-steps", default="")
    ap.add_argument("--device", default="0,1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.00005)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--amp", type=parse_bool, default=True)
    ap.add_argument("--val", type=parse_bool, default=False)
    ap.add_argument("--hsv-h", type=float, default=0.015)
    ap.add_argument("--hsv-s", type=float, default=0.60)
    ap.add_argument("--hsv-v", type=float, default=0.35)
    ap.add_argument("--degrees", type=float, default=3.0)
    ap.add_argument("--translate", type=float, default=0.12)
    ap.add_argument("--scale", type=float, default=0.60)
    ap.add_argument("--shear", type=float, default=1.0)
    ap.add_argument("--perspective", type=float, default=0.0003)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--mosaic", type=float, default=0.70)
    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--copy-paste", type=float, default=0.0)
    ap.add_argument("--erasing", type=float, default=0.0)
    ap.add_argument("--close-mosaic", type=int, default=0)
    args = ap.parse_args()

    requested_steps = parse_steps(args.checkpoint_steps)
    state = {"step": 0}

    def save_midstep_checkpoint(trainer) -> None:
        state["step"] += 1
        step = state["step"]
        if RANK != 0 or step not in requested_steps:
            return
        trainer.save_model()
        if not trainer.last.is_file():
            raise RuntimeError(f"Checkpoint file is missing after save at step {step}")
        target = trainer.wdir / f"step{step:05d}.pt"
        shutil.copy2(trainer.last, target)
        metadata = {
            "step": step,
            "epoch_zero_based": int(trainer.epoch),
            "weights": str(target),
            "source_last": str(trainer.last),
            "note": "Mid-epoch recovery checkpoint; load as initialization if recovery is needed.",
        }
        (trainer.wdir / f"step{step:05d}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info(f"[midstep-checkpoint] saved {target}")

    model = YOLO(args.model)
    model.add_callback("on_train_batch_end", save_midstep_checkpoint)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=False,
        cache=False,
        seed=args.seed,
        patience=0,
        optimizer=args.optimizer,
        lr0=args.lr0,
        amp=args.amp,
        val=args.val,
        plots=False,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        close_mosaic=args.close_mosaic,
    )


if __name__ == "__main__":
    main()
