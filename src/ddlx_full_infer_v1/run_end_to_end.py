from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .model import DDLXTrack3Model


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def python_cmd(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single DDL-X Track 3 model package from images to JSON.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gpus", default="auto", help="auto, cpu, or a CUDA_VISIBLE_DEVICES string such as 0,1,2")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-explain-shards", type=int, default=1)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--detector-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=2048)
    parser.add_argument("--swift-command", default="swift", help="swift executable used for Qwen explanation inference")
    parser.add_argument("--skip-qwen", action="store_true", help="Produce base JSON/zip without rerunning Qwen; for smoke tests only.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def gpu_env(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    if gpus == "auto":
        return env
    if gpus == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DDLXTrack3Model(args.model_root.expanduser().resolve())
    model.validate_files(require_qwen=not args.skip_qwen)
    env = gpu_env(args.gpus)

    manifests = out_dir / "manifests"
    preprocess = out_dir / "preprocess"
    cls_dir = out_dir / "classification"
    det_dir = out_dir / "detectors"
    base_json = out_dir / "base_json"
    final_json = out_dir / "final_json"
    explain_inputs = out_dir / "explain_inputs"
    qwen_predictions = out_dir / "qwen_predictions"

    run(
        python_cmd(
            "src.ddlx_full_infer_v1.prepare_test_images",
            "--image-dir",
            str(args.image_dir),
            "--out-dir",
            str(manifests),
            "--num-shards",
            str(args.num_shards),
        ),
        cwd=repo,
        env=env,
    )
    test_images = manifests / "test_images.csv"

    run(
        python_cmd(
            "src.ddlx_full_infer_v1.preprocess_test_faces",
            "--test-images",
            str(test_images),
            "--out-dir",
            str(preprocess),
            "--device",
            "auto",
        ),
        cwd=repo,
        env=env,
    )
    face_manifest = preprocess / "face_manifest_accepted.csv"

    run(
        python_cmd(
            "src.ddli_classification_v1.predict_test_faces",
            "--face-manifest",
            str(face_manifest),
            "--test-images",
            str(test_images),
            "--checkpoint",
            str(model.classifier_checkpoint),
            "--predictions-out",
            str(cls_dir / "face_predictions.csv"),
            "--image-scores-out",
            str(cls_dir / "image_scores.csv"),
            "--batch-size",
            str(args.classifier_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--amp",
        ),
        cwd=repo,
        env=env,
    )

    detector_device = "cpu" if args.gpus == "cpu" else "0"
    detector_outputs: dict[str, Path] = {}
    for name, ckpt in model.detector_checkpoints.items():
        out_csv = det_dir / f"{name}_detections.csv"
        detector_outputs[name] = out_csv
        run(
            python_cmd(
                "src.ddli_detector_v1.predict_yolo_test_faces",
                "--manifest",
                str(face_manifest),
                "--model",
                str(ckpt),
                "--out-csv",
                str(out_csv),
                "--batch",
                str(args.detector_batch_size),
                "--device",
                detector_device,
            ),
            cwd=repo,
            env=env,
            )

    wbf_json = det_dir / "detector_pred_boxes.json"
    detector_a = "detector_a_fullmask_stageb"
    detector_b = "detector_b_conservative_stageb"
    detector_c = "detector_c_yolov8m_stageb"
    run(
        python_cmd(
            "src.ddli_detector_v1.merge_wbf_test_boxes_hetero_v1",
            "--detector-a-glob",
            str(detector_outputs[detector_a]),
            "--detector-b-glob",
            str(detector_outputs[detector_b]),
            "--detector-c-glob",
            str(detector_outputs[detector_c]),
            "--pre-conf",
            "0.125",
            "--wbf-iou",
            "0.35",
            "--post-conf",
            "0.175",
            "--max-boxes",
            "3",
            "--require-models",
            "2",
            "--out-json",
            str(wbf_json),
            "--out-summary",
            str(det_dir / "wbf_summary.json"),
        ),
        cwd=repo,
        env=env,
    )

    run(
        python_cmd(
            "src.ddlx_full_infer_v1.build_base_submission_json",
            "--test-images",
            str(test_images),
            "--image-scores",
            str(cls_dir / "image_scores.csv"),
            "--detector-boxes",
            str(wbf_json),
            "--face-predictions",
            str(cls_dir / "face_predictions.csv"),
            "--raw-face-dir",
            str(preprocess / "raw_face_outputs"),
            "--output-json-dir",
            str(base_json),
            "--summary-path",
            str(out_dir / "base_json_summary.json"),
            "--zip-path",
            str(out_dir / "base_submission_without_qwen.zip"),
        ),
        cwd=repo,
        env=env,
    )

    if args.skip_qwen:
        if final_json.exists():
            shutil.rmtree(final_json)
        shutil.copytree(base_json, final_json)
        final_zip = out_dir / "submission_model_rerun.zip"
        run(
            python_cmd(
                "src.ddlx_full_infer_v1.build_base_submission_json",
                "--test-images",
                str(test_images),
                "--image-scores",
                str(cls_dir / "image_scores.csv"),
                "--detector-boxes",
                str(wbf_json),
                "--face-predictions",
                str(cls_dir / "face_predictions.csv"),
                "--raw-face-dir",
                str(preprocess / "raw_face_outputs"),
                "--output-json-dir",
                str(final_json),
                "--summary-path",
                str(out_dir / "run_summary.json"),
                "--zip-path",
                str(final_zip),
            ),
            cwd=repo,
            env=env,
        )
    else:
        run(
            python_cmd(
                "src.ddlx_full_infer_v1.build_explain_shards",
                "--submission-json-dir",
                str(base_json),
                "--test-images",
                str(test_images),
                "--out-dir",
                str(explain_inputs),
                "--num-shards",
                str(args.num_explain_shards),
            ),
            cwd=repo,
            env=env,
        )
        qwen_predictions.mkdir(parents=True, exist_ok=True)
        for shard in range(args.num_explain_shards):
            shard_name = f"shard_{shard:02d}"
            run(
                [
                    args.swift_command,
                    "infer",
                    "--model",
                    str(model.qwen_base),
                    "--adapters",
                    str(model.qwen_lora),
                    "--template",
                    "qwen2_5_vl",
                    "--val_dataset",
                    str(explain_inputs / f"{shard_name}.jsonl"),
                    "--result_path",
                    str(qwen_predictions / f"{shard_name}_predictions.jsonl"),
                    "--max_new_tokens",
                    str(args.qwen_max_new_tokens),
                    "--max_length",
                    "4096",
                    "--max_pixels",
                    "602112",
                    "--temperature",
                    "0",
                    "--num_beams",
                    "1",
                ],
                cwd=repo,
                env=env,
            )
        run(
            python_cmd(
                "src.ddlx_full_infer_v1.merge_explanations",
                "--source-json-dir",
                str(base_json),
                "--manifest-dir",
                str(explain_inputs),
                "--prediction-dir",
                str(qwen_predictions),
                "--output-json-dir",
                str(final_json),
                "--zip-path",
                str(out_dir / "submission_model_rerun.zip"),
                "--summary-path",
                str(out_dir / "run_summary.json"),
                "--num-shards",
                str(args.num_explain_shards),
            ),
            cwd=repo,
            env=env,
        )

    summary = {
        "mode": "single_model_package_end_to_end",
        "image_dir": str(args.image_dir),
        "model_root": str(model.model_root),
        "out_dir": str(out_dir),
        "skip_qwen": bool(args.skip_qwen),
        "final_zip": str(out_dir / "submission_model_rerun.zip"),
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
