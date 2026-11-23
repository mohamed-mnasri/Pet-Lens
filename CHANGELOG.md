# Changelog

## 1.0.0 (unreleased)

Phase 4: live and presentation.

- Live camera page with per-animal smoothing and measured FPS (Step 16).
- `make up && make demo`: a seeded demo database and the page, with no dataset or GPU (Step 17).
- Generated `MODEL_CARD.md`, `DATASET_CARD.md` and `ATTRIBUTION.md` (`make cards`).
- README with results, architecture, data model and dbt lineage; ADRs 0002 to 0008; engineering write-up.

## 0.4.0

Phase 3: VLM and pets.

- VLM image screener with `qwen3-vl:4b` (Ollama), measured against my reviews (Step 13).
- VLM breed-naming experiment (Step 14).
- Pet profiles: matching by appearance, naming, merge and split, append-only identity events (Step 15).

## 0.3.0

Phase 2: data engineering and feedback loop.

- Swipe review app with blind audit and reviewer checks (Step 9).
- Evaluation module and dbt marts with contracts, snapshots and unit tests (Step 10).
- Model registry, frozen golden set and paired promotion gate (Step 11).
- Partial fine-tuning and backbone ablation (Step 12).

## 0.2.0

Phase 1: core.

- Dataset pipeline with grouped splits and held-out sets (Step 2).
- DINOv2 embeddings, calibrated heads, open-set score and decision policy (Steps 3 and 4).
- Detection, person blurring and clip grouping (Step 5).
- Postgres schema, API with input limits, job queue and worker, upload page with evidence (Steps 6 to 8).
