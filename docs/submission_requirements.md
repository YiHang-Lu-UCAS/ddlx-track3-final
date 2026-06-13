# Workshop Paper Submission Requirements

## Source

User-provided organizer email, dated in conversation context:

- Award-eligible teams are required to submit a workshop paper.
- Deadline: June 28, 2026, UTC+08:00.
- Submission link: https://chairingtool.com/conferences/ddl20-ijcai2026/main-track?role=author
- Main body: up to 7 pages.
- References: excluded from the page limit.
- Formatting: follow IJCAI-ECAI 2026 main track formatting guidelines.
- Required dataset citations:
  - MFFI: Multi-dimensional face forgery image dataset for real-world scenarios
  - DDL: A large-scale datasets for deepfake detection and localization in diversified real-world scenarios

## Official Formatting Links

- IJCAI Author Kit: https://www.ijcai.org/authors_kit
- IJCAI-ECAI 2026 Main Track CFP: https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-main-track/

## Paper-Level Hard Constraints

- The paper must be a DDL-X Track 3 challenge system paper.
- The main body must fit within 7 pages in IJCAI-ECAI 2026 style.
- References must include MFFI and DDL.
- The body must mention MFFI and DDL, not only list them in the bibliography.
- The paper must not claim a new general SOTA unless directly supported by evidence.
- The paper must distinguish:
  - exact artifact verification: cached Qwen-generated text zips plus saved WBF boxes;
  - method-level reproduction: Qwen2.5-VL-3B-Instruct + LoRA checkpoint-1500 inference from image, label, and boxes.

## Facts To Keep Separate

- Historical final artifact text cache:
  - Final leaderboard packaging reused Qwen-generated text zips for speed and exact hash reconstruction.
- Method-level explanation reproduction:
  - The verification bundle includes Qwen2.5-VL-3B-Instruct, LoRA checkpoint-1500, prompt construction scripts, and inference scripts.
  - The method-level reproduction scripts default to QWEN_MAX_NEW_TOKENS=2048, the largest verified setting in the progressive rerun workflow.
- BERTScore evaluation:
  - The BERTScore summary selected `mean_f1` as the metric.
  - The Qwen evaluation summary records checkpoint-1500 as clearly better than the fixed template baseline.
  - The BERTScore tuning/evaluation settings should not be conflated with the later method-level full-regeneration setting.
