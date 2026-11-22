# PetLens: build plan

Working roadmap for PetLens. I update this file as steps land. The full design rationale lives in the README; this file is about order of work, scope per step and what "done" means.

**Current step:** 1. Local setup and repo skeleton
**Last updated:** 2026-11-22

## Principles I'm holding myself to

- **Phases 1 and 2 before anything in Phase 3.** A working feedback loop matters more than extra features.
- **Every step ends with something that runs** and a check I can point to. No step is done on "mostly works".
- **Real footage from week 1.** I'm collecting photos and clips of real animals in parallel with the code, not at the end.
- **Honest numbers only.** Whatever the README reports must come from the code and the dbt marts in this repo.
- **Nothing large in git.** Datasets, weights, footage and the database live outside the repo.

## Status

| # | Step | Phase | Status |
| --- | --- | --- | --- |
| 1 | Local setup and repo skeleton | 1: Core | In progress |
| 2 | Dataset download and split manifest | 1: Core | Not started |
| 3 | Embeddings, baseline head, calibration, ONNX export | 1: Core | Not started |
| 4 | Unknown-breed detection and decision policy | 1: Core | Not started |
| 5 | Detection, tracking and person blurring | 1: Core | Not started |
| 6 | Database schema, contracts and API | 1: Core | Not started |
| 7 | Live and Upload pages (Phase 1 gate) | 1: Core | Not started |
| 8 | Review queue, blind audit and quality signals | 2: Data engineering | Not started |
| 9 | dbt evaluation marts and tests | 2: Data engineering | Not started |
| 10 | Model registry, golden set and promotion gate | 2: Data engineering | Not started |
| 11 | GPU fine-tuning and regression gate (Phase 2 gate) | 2: Data engineering | Not started |
| 12 | AI auditor worker | 3: Auditor and profiles | Not started |
| 13 | Agreement routing and auditor scorecard | 3: Auditor and profiles | Not started |
| 14 | Error predictor | 3: Auditor and profiles | Not started |
| 15 | Pet profiles and re-identification (Phase 3 gate) | 3: Auditor and profiles | Not started |
| 16 | One-command demo, README and ADRs | 4: Presentation | Not started |
| 17 | Demo video and write-up | 4: Presentation | Not started |

Rough timeline: 10 to 12 weeks at 10 to 12 hours a week. Phase 1 is weeks 1 to 3, Phase 2 weeks 4 to 6, Phase 3 weeks 7 to 9, Phase 4 weeks 10 to 12.

## Machine and layout

Development machine: Windows 11 laptop, i7-10750H, 32 GB RAM, RTX 2060 (6 GB VRAM), 720p USB webcam.

| What | Where | Notes |
| --- | --- | --- |
| Code | `C:\Users\Mohamed\Desktop\PROJECTS\petlens` | Excluded from OneDrive sync |
| Datasets, weights, footage | `D:\petlens-data\` (`raw`, `processed`, `models`, `footage`) | Never committed |
| Postgres data | Docker named volume | Docker Desktop disk image moved to D: |
| Python env | `.venv` in the repo, managed by `uv` | Git-ignored |

The Python app, camera capture and Ollama run natively on Windows, because Docker can't reach a USB webcam easily. Only Postgres runs in Docker. CI runs on Linux. See [ADR 0001](adr/0001-local-first-windows-layout.md).

## Step 1: Local setup and repo skeleton

**Goal:** a clean, reproducible base before any ML code. After this step every later step is just adding modules.

**Done when:**

- [ ] `make check` passes locally (Ruff, mypy, pytest)
- [ ] `make db-up` starts Postgres 16 with the `vector` extension, and the DB smoke test passes against it
- [ ] `nvidia-smi` shows the RTX 2060 and its real VRAM
- [ ] CI is green on GitHub Actions, including the DB smoke test against a service container

### Tasks

**1.1 Tooling.** Installed through winget: Git, uv, Docker Desktop (WSL 2 engine), VS Code, GNU Make (`ezwinports.make`). Python 3.12 through `uv python install 3.12`. NVIDIA driver updated. Ollama waits until Step 12.

**1.2 Repo.** Cloned into the projects folder. `.gitattributes` forces LF line endings so Windows and Linux CI agree. Data folders created on D:.

**1.3 Package layout.** `src/` layout, one package:

```
src/petlens/
  config.py   settings from env / .env (DATABASE_URL, DATA_DIR)
  data/       step 2
  models/     steps 3-4
  vision/     step 5
  db/         step 6
  api/        step 6
  ui/         step 7
tests/
docs/adr/
```

**1.4 Quality tooling.** Ruff for lint and format, mypy in strict mode on `src/`, pytest, pre-commit (Ruff, whitespace fixes, a large-file guard so no image or weight file gets committed by accident).

**1.5 Database.** `docker-compose.yml` runs `pgvector/pgvector:pg16` with a healthcheck. An init script enables the `vector` extension. Connection settings come from `.env` (template in `.env.example`).

**1.6 Makefile.** `install`, `lint`, `format`, `types`, `test`, `check`, `db-up`, `db-down`. Later steps add `data`, `train`, `up`, `demo`.

**1.7 CI.** GitHub Actions on push and pull request: uv, Ruff, mypy, pytest with a Postgres + pgvector service container. CI never downloads datasets or model weights.

**1.8 Close out.** ADR 0001 written. PR merged when green, tagged `v0.1.0`. First batch of 10 to 20 clips and photos of real cats and dogs saved to `D:\petlens-data\footage`.

## Steps 2 to 17

### Phase 1: Core

**Step 2. Dataset and splits.** `make data` downloads Oxford-IIIT Pet to D:, verifies checksums and writes a fixed-seed train / val / calibration / test split manifest. Four breeds are held out entirely to test unknown-breed detection later. The manifest is committed; images are not.

**Step 3. Baseline classifier.** PyTorch with CUDA. DINOv2 ViT-S/14 embeddings for every image, stored as Parquet. Logistic-regression head, then a small MLP, picked on validation macro-F1. Temperature scaling on the calibration split. ONNX export.
Done when: test top-1, macro-F1 and ECE before and after calibration are recorded.

**Step 4. Unknown breeds and decision policy.** kNN or Mahalanobis distance in embedding space, tuned on the held-out breeds and non-pet images. Decision policy (accept / abstain / review) as plain config.
Done when: every policy branch has a unit test, and held-out breeds are mostly flagged as unknown.

**Step 5. Detection and tracking.** RF-DETR (Apache-licensed sizes only) for cats and dogs, ByteTrack via `supervision`, track-level voting so video doesn't flicker, people blurred.
Done when: a video goes in and tracked boxes with per-track top 3 come out, with webcam FPS measured.

**Step 6. Schema, contracts, API.** Alembic migrations for the core tables, Pydantic contracts mirroring them, a dead-letter table for rejected payloads. FastAPI endpoints: `/predict/image`, `/predict/video`, `/runs/{id}`.
Done when: every event lands in Postgres and API tests run in CI.

**Step 7. Live and Upload pages.** Streamlit, camera view with tracked boxes and top-3 cards, upload grid.
**Phase 1 gate:** image, video or camera in; calibrated top 3 and a decision out; every event in Postgres; CI green. Tag `v0.2.0`.

### Phase 2: Data engineering and feedback loop

**Step 8. Review queue.** Review page and `review` table with `sample_reason`. 5% of confident predictions go to a blind random audit so accuracy isn't only measured on the hard cases. Quality signals: blur, brightness, box size.

**Step 9. dbt marts.** Staging, intermediate, and marts for accuracy, calibration, confused pairs, failure slices and abstention. dbt tests and unit tests on the calibration logic.
Done when: `dbt build` is green locally and in CI on fixtures.

**Step 10. Registry and promotion.** `model_registry` table, frozen golden set, champion vs challenger script. Model Health page reads only from marts.

**Step 11. Fine-tuning.** Unfreeze the last DINOv2 blocks on the GPU (fp16), compare ViT-S and ViT-B, add a CI regression gate on a tiny fixture model.
**Phase 2 gate:** after 100+ reviews, v2 vs v1 on the audit sample, shown from dbt. Tag `v0.3.0`.

### Phase 3: AI auditor and pet profiles

**Step 12. Auditor worker.** Ollama with `qwen3-vl:4b` (local tags only). Background job after each session or upload, never during live view because of the 6 GB VRAM limit. Versioned prompts, JSON output validated with Pydantic.

**Step 13. Routing and scorecard.** Disagreements go to Review with both answers shown. `mart_auditor_scorecard` compares classifier, auditor, both-agree and a plain confidence-threshold baseline. `make eval-auditor` for prompt changes.

**Step 14. Error predictor.** scikit-learn model that predicts which predictions are wrong, compared with threshold-only and auditor-only routing at a fixed review budget.

**Step 15. Pet profiles.** pgvector matching plus track continuity, name / merge / split with SCD2 history, Pets page, LLM profile cards checked fact by fact, re-ID precision and recall on my own labelled photos.
**Phase 3 gate:** 300+ reviewed predictions answer "does the auditor beat the threshold?", and my pets group correctly. Tag `v0.4.0`.

### Phase 4: Presentation

**Step 16.** `make up && make demo` from a fresh clone with a seeded demo database. README with results, diagrams, ERD, dbt lineage and limitations. Five or six ADRs.

**Step 17.** 45-second demo video, write-up, tag `v1.0.0`.

## If time runs short

Cut in this order: training-data audit, gemma comparison, profile cards.
Not cutting: unknown-breed handling, review with blind audit, auditor scorecard, re-ID, tests.

## Open decisions

- **Licence:** decide before the repo goes public. Leaning MIT or Apache 2.0. Oxford-IIIT images stay CC BY-SA and are never in the repo.
- **Going public:** planned at the Phase 2 gate, earlier if Phase 1 looks presentable.
- **Python version:** pinned to 3.12 for the widest PyTorch, RF-DETR and dbt support.
