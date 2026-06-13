# Reproduce / Verification Notes

This bundle supports two verification levels.

## Level 1: exact submitted zip verification

Run:

```bash
bash scripts/verify_bundle.sh
```

This checks that the copied final zip matches the highest-scoring server artifact:

```text
a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97
```

## Level 2: rebuild submission zips from saved WBF boxes

Given the DDL-X metadata directory used by the original pipeline, run:

```bash
bash scripts/rebuild_from_saved_wbf_boxes.sh   /media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_clsadapt020_adapt384_cleanup_top3_v1   /media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_fast/face_shard_outputs   /tmp/ddlx_rebuild_wbf
```

This uses:

- `evidence/detector_pred_boxes.json`
- copied text source zips in `text_sources/`
- `src/ddli_detector_v1/build_wbf_text_from_variant_zips.py`

The selected variant after rebuild is `fake_nobox_nose_eyes_mouth`.

## Full detector rerun

The trained weights and WBF code are included for result verification. The original full-test launcher is preserved as:

```text
scripts/launch_wbf_hetero_fulltest_4variants_v1.sh
```

That launcher records the original absolute server paths and the exact GPU execution layout used during the challenge run. It is kept as provenance; adapt the paths if running in a clean verification environment.


## Level 3: Qwen explanation model verification

The bundle includes the Qwen explanation model and LoRA adapter:

```text
models/explanation/qwen2_5_vl_3b_instruct/
models/explanation/qwen2_5_vl_3b_lora_checkpoint1500/
```

The original inference commands are preserved in:

```text
src/ddli_explain_v1/launch_test_explain_full_6gpu_v1.sh
src/ddli_explain_v1/launch_repeat2_fake_nobox_text_rerun_5variants_6gpu_v1.sh
```

They use `swift infer` with:

```text
--model Qwen2.5-VL-3B-Instruct
--adapters checkpoint-1500
--template qwen2_5_vl
```

The final submitted zip itself is rebuilt from already-generated explanation zips in `text_sources/`, because those are the exact text artifacts used by the highest-scoring WBF package.


## Recommended explanation verification procedure

To verify that text explanations are model-generated, run `scripts/run_explanation_inference.sh` on a small image subset or on the full test metadata if available. This directly invokes the bundled Qwen model and LoRA adapter. The cached text zips under `text_sources/` should only be used when checking exact hash equivalence of the previously submitted leaderboard zip.
