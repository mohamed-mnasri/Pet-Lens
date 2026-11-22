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
| 2 | Dataset pipeline: Oxford + Wikimedia Commons, grouped splits | 1: Core | Not started |
| 3 | Embeddings, baseline head, calibration, Commons screening | 1: Core | Not started |
| 4 | Unknown breeds, domestic cats and decision policy | 1: Core | Not started |
| 5 | Detection, tracking and person blurring | 1: Core | Not started |
| 6 | Database schema, contracts, API and input limits | 1: Core | Not started |
| 7 | Live and Upload pages (Phase 1 gate) | 1: Core | Not started |
| 8 | Human-in-the-loop batch validation, review queue and blind audit | 2: Data engineering | Not started |
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

**Step 2. Dataset pipeline and splits.** `make data` builds a versioned manifest from two sources. The full reasoning, numbers and sources are in [DATASET.md](DATASET.md).

- **Oxford-IIIT Pet** (core, clean labels): download, verify checksums, derive body crops from the trimaps, keep the official test split minus leaked duplicates.
- **Wikimedia Commons** (expansion and unseen-source test): pull breed categories through the MediaWiki API with a proper User-Agent and throttling. Keep only CC0, public domain, CC BY and CC BY-SA files, and record author, licence and URL for every file. Commons category names are weak labels, not ground truth.
- **Filters:** exactly one cat or dog found by the detector, short side at least 224 px, corrupt and near-black files dropped.
- **Duplicates:** SHA-256 for exact copies, pHash plus DINOv2 cosine similarity for near-duplicates, grouped into clusters.
- **Splits:** grouped by cluster, cat, video or Commons uploader, stratified, fixed seed. Existing assignments never move when data is added. A set of Commons uploaders is held out entirely as the unseen-source test.
- **Held-out breeds:** 2 cat and 2 dog breeds, kept out of training to test unknown-breed detection.
- Manifests, `breeds.yaml`, `sources.yaml` and a quality report are committed; images are not.

Done when: `make data` reruns with no changes, a test proves no group crosses splits, every row has a licence and attribution, and the quality report shows counts per class, source and split.

**Step 3. Baseline classifier and Commons screening.** PyTorch with CUDA. DINOv2 ViT-S/14 embeddings for every crop, stored as Parquet. Per-species logistic-regression head (12 cat, 25 dog breeds), then a small MLP, picked on validation macro-F1. Temperature scaling on the calibration split. ONNX export.
v0 trains on Oxford only, then screens the Commons candidates: if the category label is in the model's top 3, the image joins the training pool; if not, it goes to my review queue in Step 8 instead of being dropped. A random 5% of the agreements also go to review, so I can measure how noisy the agreed labels are.
Done when: top-1, top-3, macro-F1 and ECE before and after calibration are recorded on both the in-domain test and the unseen-source test, and a learning curve (25 / 50 / 100% of training data) is plotted.

**Step 4. Unknown breeds, domestic cats and decision policy.** kNN or Mahalanobis distance in embedding space, tuned on the held-out breeds and non-pet images. Decision policy (accept / abstain / review) as plain config. A cat that doesn't clearly match a breed gets "domestic cat, no clear breed", which is the right answer for most pet cats, not a failure. Explicit domestic shorthair and longhair classes come later, once each has 150+ checked images from 3+ sources.
Done when: every policy branch has a unit test, held-out breeds are mostly flagged as unknown, and the risk-coverage curve and abstention rate on domestic cats are reported.

**Step 5. Detection and tracking.** RF-DETR (Apache-licensed sizes only) for cats and dogs, ByteTrack via `supervision`, track-level voting so video doesn't flicker, people blurred.
Done when: a video goes in and tracked boxes with per-track top 3 come out, with webcam FPS measured.

**Step 6. Schema, contracts, API.** Alembic migrations for the core tables, Pydantic contracts mirroring them, a dead-letter table for rejected payloads. FastAPI endpoints: `/predict/image`, `/predict/video`, `/runs/{id}`.

Input limits, enforced at the API before anything is decoded or stored (values live in config):

| Input | Limit | What gets stored |
| --- | --- | --- |
| Photo | JPEG, PNG or WebP; at most 15 MB and 40 megapixels | A copy resized to 1280 px on the long side (JPEG, quality 90), plus one 448 px crop per detected animal |
| Video | MP4, MOV or WebM; at most 250 MB and 3 minutes; decoded at up to 1080p | Sampled at 5 fps for detection and tracking; the 3 sharpest 448 px crops per track. The original is deleted after processing unless it is in a review batch |
| Live camera | No recording | The sharpest crop per track, at most one per track per minute |
| Review batch | At most 200 files | Originals kept for 30 days for review, then deleted; crops and labels stay |

Over-limit files are rejected with a clear message and logged to the dead-letter table, never silently trimmed. `make storage-report` shows usage per folder, and unreviewed media older than the retention window is pruned. Rough budget: a 448 px crop is about 40 KB and a ViT-S embedding is 1.5 KB, so 10,000 reviewed crops come to under 500 MB.

Done when: every event lands in Postgres, API tests run in CI, and every limit has a test with an over-limit file.

**Step 7. Live and Upload pages.** Streamlit, camera view with tracked boxes and top-3 cards, upload grid. The upload page shows the limits up front and rejects oversize files before uploading them.
**Phase 1 gate:** image, video or camera in; calibrated top 3 and a decision out; every event in Postgres; CI green. Tag `v0.2.0`.

### Phase 2: Data engineering and feedback loop

**Step 8. Human-in-the-loop validation.** This is where the model gets checked by a person and the checked results flow back into the data.

- **Batch validation:** I drop a batch of photos and videos into `D:\petlens-data\inbox\<batch>`. `make review-batch` runs them through the full pipeline (detect, crop, classify, decide) and queues every prediction.
- **Review page:** for each crop I click *right*, *wrong* or *not sure*. On *wrong* I pick from the top 3, "domestic / no clear breed", "not a cat or dog", or type the breed. *Not sure* is a valid answer; I'm not a breed expert and guessing would add noise.
- **Commons queue:** the Commons images Step 3 couldn't agree with, plus the 5% random sample of agreements, go through the same page.
- **Blind audit:** 5% of confident predictions are sampled at random, so accuracy isn't only measured on the hard cases.
- **Storage:** predictions and `label_event` rows are append-only. A label event records who proposed it (me, Commons category, auditor), its status and when it was reviewed. Nothing overwrites history.
- **Getting into training:** only reviewed labels enter a new dataset version, and they go through the Step 2 pipeline like any other source (dedupe, grouping, quality checks). My own cats are grouped by identity and mostly feed the test and golden sets, capped at 20 images per cat per version. The model's own unreviewed predictions never become training labels, and audit and golden-set images are never trained on.
- Quality signals: blur, brightness, box size.

Done when: a batch of 50+ real photos and clips goes in, gets reviewed, and the reviewed labels show up in the next dataset version's manifest with their origin recorded.

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

- **Licence:** decide before the repo goes public. Leaning MIT or Apache 2.0. Oxford-IIIT images stay CC BY-SA and are never in the repo. Commons images are never in the repo either; an attribution list is generated from the manifest.
- **Stanford Dogs:** optional for wider dog coverage. It comes from ImageNet, whose terms allow non-commercial research and education only, so it stays out unless I decide that's acceptable for a portfolio project.
- **Going public:** planned at the Phase 2 gate, earlier if Phase 1 looks presentable.
- **Python version:** pinned to 3.12 for the widest PyTorch, RF-DETR and dbt support.
