import argparse
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


FAKE_SUMMARY = "Summary: This image has been tampered with."
REAL_SUMMARY = "Summary: This image has not been tampered with."


def expected_summary(label):
    normalized = str(label).strip().lower()
    if normalized == "fake":
        return FAKE_SUMMARY
    if normalized == "real":
        return REAL_SUMMARY
    raise ValueError(f"unsupported Classification result: {label!r}")


def normalize_text(text, summary):
    text = "" if text is None else str(text)
    stripped = text.rstrip()
    if stripped.endswith(summary):
        return stripped, False
    separator = "\n\n" if stripped else ""
    return f"{stripped}{separator}{summary}", True


def clone_zip_info(info):
    cloned = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    cloned.compress_type = info.compress_type
    cloned.comment = info.comment
    cloned.extra = info.extra
    cloned.internal_attr = info.internal_attr
    cloned.external_attr = info.external_attr
    cloned.create_system = info.create_system
    cloned.flag_bits = info.flag_bits
    return cloned


def build_fixed_zip(source, destination):
    counts = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{destination.stem}.",
        suffix=".tmp.zip",
        dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(
            temp_path, "w", allowZip64=True
        ) as dst:
            for info in src.infolist():
                raw = src.read(info.filename)
                if not info.filename.lower().endswith(".json"):
                    dst.writestr(clone_zip_info(info), raw)
                    counts["copied_non_json"] += 1
                    continue

                counts["json_entries"] += 1
                payload = json.loads(raw)
                summary = expected_summary(payload.get("Classification result"))
                text, changed = normalize_text(
                    payload.get("Visible forgery traces"), summary
                )
                payload["Visible forgery traces"] = text
                counts["changed" if changed else "already_correct"] += 1

                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=4,
                ).encode("utf-8")
                dst.writestr(clone_zip_info(info), encoded)

        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return dict(counts)


def validate_zip(path):
    counts = Counter()
    with zipfile.ZipFile(path, "r") as archive:
        for info in archive.infolist():
            if not info.filename.lower().endswith(".json"):
                continue
            counts["json_entries"] += 1
            payload = json.loads(archive.read(info.filename))
            summary = expected_summary(payload.get("Classification result"))
            text = payload.get("Visible forgery traces")
            text = "" if text is None else str(text)
            if text.rstrip().endswith(summary):
                counts["correct_suffix"] += 1
            else:
                counts["bad_suffix"] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    sources = sorted(args.input_dir.glob("*.zip"))
    if not sources:
        raise SystemExit(f"no zip files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for source in sources:
        destination = args.output_dir / source.name.replace(
            ".zip", "_summary_suffix_fixed.zip"
        )
        build = build_fixed_zip(source, destination)
        validation = validate_zip(destination)
        item = {
            "source": str(source),
            "destination": str(destination),
            "build": build,
            "validation": validation,
            "bytes": destination.stat().st_size,
        }
        report.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    report_path = args.output_dir / "summary_suffix_validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
