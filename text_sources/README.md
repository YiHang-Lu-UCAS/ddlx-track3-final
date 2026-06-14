# Text Sources

The final leaderboard artifact is reproduced exactly from saved WBF boxes and cached Qwen-generated text zips.

Important distinction:

- `text_sources/` are cached Qwen-generated explanation artifacts used for exact zip/hash reconstruction.
- They are not the explanation method itself.
- Method-level explanation reproduction uses Qwen2.5-VL-3B-Instruct, the LoRA checkpoint-1500 adapter, and the scripts in `scripts/run_explanation_inference.sh` / `src/ddli_explain_v1/`.

The actual text source zip files are large and are hosted in the Hugging Face asset repository or included in the organizer verification bundle.
