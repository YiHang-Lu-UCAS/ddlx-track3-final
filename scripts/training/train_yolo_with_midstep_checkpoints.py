from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER, RANK


def parse_steps(raw: str) -> set[int]:
    return {int(value) for value in raw.split(",") if value.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint-steps", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()

    requested_steps = parse_steps(args.checkpoint_steps)
    state = {"step": 0}

    def save_midstep_checkpoint(trainer) -> None:
        state["step"] += 1
        step = state["step"]
        if RANK != 0 or step not in requested_steps:
            return
        if not trainer.save_model():
            raise RuntimeError(f"Checkpoint save was rejected at step {step}")
        target = trainer.wdir / f"step{step:05d}.pt"
        shutil.copy2(trainer.last, target)
        metadata = {
            "step": step,
            "epoch_zero_based": int(trainer.epoch),
            "weights": str(target),
            "source_last": str(trainer.last),
            "note": "Mid-epoch recovery checkpoint; load as initialization if recovery is needed.",
        }
        (trainer.wdir / f"step{step:05d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info(f"[midstep-checkpoint] saved {target}")

    model = YOLO(args.model)
    model.add_callback("on_train_batch_end", save_midstep_checkpoint)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=1,
        batch=args.batch,
        device="0,1,2,3,4,5,6,7",
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=False,
        cache=False,
        seed=args.seed,
        patience=0,
        amp=True,
        val=False,
        plots=False,
    )


if __name__ == "__main__":
    main()
