# PetLens: build plan

Working roadmap for PetLens. I update this file as steps land. The design itself lives in [SPEC.md](SPEC.md) and the data choices in [DATASET.md](DATASET.md); this file is about order of work, scope per step and what "done" means.

**Current step:** 3. Embeddings and baseline classifier
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
| 1 | Local setup and repo skeleton | 1: Core | In review |
| 2 | Dataset pipeline: Oxford + Commons, domestic and unknown sets, grouped splits | 1: Core | Done (first manifest: 2026-11-22-2e986370) |
| 3 | Embeddings, baseline heads, calibration, Commons screening | 1: Core | In progress (3a done) |
| 4 | Open-set score and decision policy | 1: Core | In progress (code done, first tuned run pending) |
| 5 | Detection, person blurring and clip grouping by appearance | 1: Core | Not started |
| 6 | Database schema, contracts, API, input limits and hardening | 1: Core | Not started |
| 7 | Job queue and worker | 1: Core | Not started |
| 8 | Upload page with evidence panel (Phase 1 gate) | 1: Core | Not started |
| 9 | Swipe review app, blind audit and reviewer checks | 2: Data engineering | Not started |
| 10 | Evaluation module and dbt marts | 2: Data engineering | Not started |
| 11 | Model registry, golden set and promotion gate | 2: Data engineering | Not started |
| 12 | Fine-tuning ablations (Phase 2 gate) | 2: Data engineering | Not started |
| 13 | VLM image screener | 3: VLM and pets | Not started |
| 14 | VLM breed-naming experiment | 3: VLM and pets | Not started |
| 15 | Pet profiles (Phase 3 gate) | 3: VLM and pets | Not started |
| 16 | Live camera page | 4: Live and presentation | Not started |
| 17 | One-command demo, README, model and dataset cards, ADRs | 4: Live and presentation | Not started |
| 18 | Demo video and write-up | 4: Live and presentation | Not started |

No fixed deadline: I'm building it with AI coding help, so scope is set by quality, not hours. Each phase ends at a gate with a tagged release, so there is always a working version.

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

**1.1 Tooling.** Installed through winget: Git, uv, Docker Desktop (WSL 2 engine), VS Code, GNU Make (`ezwinports.make`). Python 3.12 through `uv python install 3.12`. NVIDIA driver updated. Ollama waits until Step 13.

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
  jobs/       step 7 (package added then)
  ui/         step 8
tests/
docs/adr/
```

**1.4 Quality tooling.** Ruff for lint and format, mypy in strict mode on `src/`, pytest, pre-commit (Ruff, whitespace fixes, a large-file guard so no image or weight file gets committed by accident).

**1.5 Database.** `docker-compose.yml` runs `pgvector/pgvector:pg16` with a healthcheck. An init script enables the `vector` extension. Connection settings come from `.env` (template in `.env.example`).

**1.6 Makefile.** `install`, `lint`, `format`, `types`, `test`, `check`, `db-up`, `db-down`. Later steps add `data`, `train`, `up`, `demo`.

**1.7 CI.** GitHub Actions on push and pull request: uv, Ruff, mypy, pytest with a Postgres + pgvector service container. CI never downloads datasets or model weights.

**1.8 Close out.** ADR 0001 written. PR merged when green, tagged `v0.1.0`. First batch of 10 to 20 clips and photos of real cats and dogs saved to `D:\petlens-data\footage`.

## Steps 2 to 18

### Phase 1: Core

**Step 2. Dataset pipeline and splits.** `make data` builds a versioned manifest. Numbers and sources are in [DATASET.md](DATASET.md).

- **Oxford-IIIT Pet** (core, clean labels): download, verify checksums, derive body crops from the trimaps, keep the official test split minus leaked duplicates. All 37 breeds stay in training.
- **Wikimedia Commons, cats only for now:**
  - Fetch the 12 Oxford cat breeds, the domestic short-haired and long-haired categories, and the non-Oxford breeds (Norwegian Forest Cat, Scottish Fold, Devon Rex, Exotic Shorthair).
  - Use the MediaWiki API with a proper User-Agent, throttling and caching.
  - Keep only CC0, public domain, CC BY and CC BY-SA files, with author, licence and URL recorded per file.
  - Category names are weak labels, not ground truth.
- **Domestic cat class:** built from the domestic categories. Every pedigree breed also gets Commons images, so the model can't learn "Oxford-style photo means pedigree".
- **Unknown-breed set:** the non-Oxford Commons breeds plus non-pet images. Never trained on.
- **Filters:** exactly one cat or dog found by the detector, short side at least 224 px, corrupt and near-black files dropped.
- **Duplicates:** SHA-256 for exact copies, pHash plus DINOv2 cosine similarity for near-duplicates, grouped into clusters.
- **Splits:**
  - Grouped by cat, clip, cluster or Commons uploader. Stratified, with a fixed seed. Existing assignments never move.
  - Held-out uploaders form the unseen-source test and, separately, a real-world calibration slice.
  - Domestic and unknown-breed test sets are kept apart.
- Manifests, `breeds.yaml`, `sources.yaml` and a quality report are committed. Images are not.

Step 2 is too big for one PR, so it lands in three parts:

| Part | Scope | Status |
| --- | --- | --- |
| 2a | `breeds.yaml`, `sources.yaml`, checksum-verified downloads, the shared ingest step (EXIF fix, metadata stripped, 1024 px WebP under 250 KB, named by SHA-256), Oxford manifest with trimap body boxes, exact-duplicate removal, ingest report. `make data-oxford` | Done |
| 2b | Commons fetcher for the cat categories: walks each category and one level of subcategories (art, toy and drawing subcategories skipped), MediaWiki API with User-Agent, throttling, `maxlag`, backoff and an on-disk cache, licence allowlist, author, licence and uploader per file, files under two labels dropped, only the 1024 px thumbnail downloaded. `make data-commons` | Done |
| 2c | pHash near-duplicate clusters, Oxford train/test leaks and Commons copies of test images excluded, groups (near-duplicates joined by uploader), hash-based splits that never move, dataset version (date + content hash, unchanged rebuilds keep the version), committed manifest and report, image checksums. `make data-build` | Done |

DINOv2 near-duplicates need the embeddings from Step 3, so they run there as a leak check: a held-out image whose embedding is a near-copy of a training image gets excluded. Splits are never moved to fix a leak.

The "exactly one cat or dog" detector filter needs the detector from Step 5. Until then Oxford uses its trimap boxes and Commons images are filtered by size and quality only; the detector filter is added as a pipeline stage in Step 5.

Done when: `make data` reruns with no changes, a test proves no group crosses splits, every row has a licence and attribution, and the quality report shows counts per class, source and split.

**Step 3. Baseline classifier and Commons screening.**

- PyTorch with CUDA. DINOv2 ViT-S/14 embeddings for every crop, stored as Parquet.
- Per-species heads: 12 cat breeds plus `domestic_cat`, and 25 dog breeds. Logistic regression first, then a small MLP, picked on validation macro-F1.
- Temperature scaling on the calibration split.
- **Screening.** v0 trains on Oxford plus the domestic set, then screens the Commons candidates. If its top-1 matches the category, the image joins the training pool. If not, it goes to my swipe queue in Step 9. A random 5% of the matches also go to review, so I can measure how noisy the agreed labels are.

Step 3 lands in two parts:

| Part | Scope | Status |
| --- | --- | --- |
| 3a | DINOv2 ViT-S/14 embeddings via `timm` (crop from the box with a 10% margin, padded square, 224 px, fp16 on the GPU, cached and resumable), logistic regression vs MLP heads per species picked on validation macro-F1, temperature scaling, metrics per test set before and after calibration, learning curve, run saved with weights hash. `make install-ml`, `make embed`, `make train` | Done |
| 3b | DINOv2 near-duplicate leak check (held-out images with cosine 0.97 or more to a training image are left out of evaluation), Commons screening (top-1 agreement admits, disagreements and a seeded 5% sample wait for review; results committed to `data/screening/<version>.parquet`), retrain with the admitted photos, balanced class weights. `make screen`, `make train` | In progress |

Torch and timm are an optional extra (`ml`), so CI and the data pipeline don't install them.

Done when: top-1, top-3, macro-F1 and ECE before and after calibration are recorded on the in-domain, unseen-source and domestic test sets, and a learning curve (25 / 50 / 100% of training data) is plotted.

**Step 4. Open-set score and decision policy.**

- **Open-set score:** kNN distance from a crop's embedding to the training embeddings, served from pgvector.
- **Decision policy:** accept, domestic / no clear breed, or review. It lives in plain config.
- **Thresholds:** tuned on the real-world calibration slice, not on in-domain data.
- **Close pairs** show both names when their scores are close.

Implementation notes:

- The kNN index is in memory for now (`petlens.models.openset`), with the interface the pgvector version will have in Step 6, because Postgres isn't running on my machine yet.
- Outcomes are `accept`, `no_clear_breed` (for cats, "domestic cat, no clear breed") and `review` (far from anything in training).
- `make train` tunes the thresholds on `rw_calibration` (cats) or the calibration split (dogs, which have no real-world slice yet) and writes them to `data/policy.yaml`. Tuning maximises accepted answers while keeping accepted answers at 90% accuracy and the false breed claim rate at 10% or lower.

Done when:

- every policy branch has a unit test
- the risk-coverage curve is reported
- the unknown-breed rejection rate is reported
- the **false breed claim rate on domestic cats** is reported against my target of 10% or lower (revisited once I see the baseline)

**Step 5. Detection and clip grouping.**

- RF-DETR (Apache-licensed sizes only) for cats, dogs and people. People are blurred in anything stored.
- No tracker. Clips are sampled at one frame per second, and detections across frames are grouped by appearance (DINOv2 similarity plus a box-size check). Each group gets its own averaged answer, so a two-cat clip gives two answers.
- If grouping is ambiguous, the clip returns per-frame results and says why.

Done when: a test clip with one cat gives one answer, a test clip with two cats gives two, and photo and clip latency are measured.

**Step 6. Schema, contracts, API, limits and hardening.**

- Alembic migrations for the core tables (see SPEC.md section 9), Pydantic contracts mirroring them, and a dead-letter table for rejected payloads.
- FastAPI endpoints: `/predict/image`, `/predict/video`, `/runs/{id}`, `/jobs/{id}`.

Input limits and compression are enforced at the API before anything is stored, and the values live in config. Originals are never kept: every photo and every sampled frame is resized and re-encoded on ingest, and the uploaded file is deleted straight after.

| Input | Upload limit | What gets stored |
| --- | --- | --- |
| Photo | JPEG, PNG or WebP; at most 15 MB and 40 megapixels | One copy, long edge 1024 px, WebP quality 80, hard cap 250 KB (quality steps down until it fits), plus a 384 px crop per detected animal (WebP, cap 40 KB) |
| Video | MP4, MOV or WebM; at most 25 MB and 10 seconds. Longer clips are rejected with "Trim the clip to 10 seconds or upload photos instead" | One frame per second (at most 10), each stored like a photo. No "best frame" logic: I'm responsible for uploading a short, clean clip. Animals are grouped across frames by appearance, and each group's answer is the average over its frames |
| Live camera (Step 16) | No recording | At most one frame every 10 seconds while an animal is in view, stored like a photo |
| Review batch | At most 200 files | The compressed copies above; nothing larger is kept for review |

**Hardening:**

- File types are checked by signature, not extension.
- Video duration is read with `ffprobe` before any decoding.
- EXIF orientation is applied once, and GPS is stripped.
- The API binds to 127.0.0.1.
- Inbox paths are sanitised.

Over-limit files are rejected with a clear message and logged to the dead-letter table, never silently trimmed. `make storage-report` shows usage per folder.

Expected storage, worst case at the caps: 100 photos come to at most 25 MB for the stored copies plus 4 MB for crops, so under 30 MB. A 10-second clip stores at most 10 frames, under 3 MB. Typical files should be well under the caps (*estimate*: 100 to 150 KB per 1024 px WebP). Embeddings add 1.5 KB per crop for ViT-S. So 10,000 stored images stay under 3 GB even at the caps.

Done when:

- every event lands in Postgres
- API tests run in CI
- every limit has a test with an over-limit file
- a test checks that a 40 MP photo is stored under 250 KB
- an Alembic upgrade and downgrade test passes

**Step 7. Job queue and worker.**

- A `job` table claimed with `SELECT … FOR UPDATE SKIP LOCKED`, and one worker process (`make worker`).
- Jobs have states, attempts and an error column, and stuck jobs are recovered after a timeout.
- Batch reviews, Commons screening and later VLM screening all run here. Single photos and clips stay synchronous.

Done when: tests cover claim, retry, crash recovery and no double claims, and the UI shows job progress.

**Step 8. Upload page.**

- Streamlit upload grid that shows the limits up front and rejects oversize files before uploading them.
- Result cards show the top 3 with calibrated probabilities, the decision badge, and the five most similar training images for the suggested breed (from pgvector).

**Phase 1 gate:** a photo or clip in gives a calibrated decision and evidence out, every event is in Postgres, an end-to-end smoke test with tiny fixture models runs in CI, and CI is green. Tag `v0.2.0`.

### Phase 2: Data engineering and feedback loop

**Step 9. Swipe review app.** This is where predictions get checked by me and the checked results flow back into the data.

- **Cards:**
  - Each card shows the crop, the suggested answer and 3 to 5 reference photos of that breed.
  - Right means correct, left means wrong, up means not sure. Arrow keys work too.
  - On "wrong", a quick picker offers the other top-3 answers, "domestic / no clear breed", "not a cat or dog" or "skip".
  - *Not sure* is a valid answer. I'm not a breed expert, and guessing adds noise.
- **What I'm asked:** questions I can answer reliably come first (real photo, whole animal, cat or not, domestic or clearly pedigree, which of my pets). I only confirm a fine breed when I actually know it.
- **Queues:**
  - Commons images Step 3 didn't agree with, plus the 5% sample of agreements
  - my own batches (`make review-batch` drops a folder from `D:\petlens-data\inbox\<batch>` into the job queue)
  - a blind audit of 5% of confident predictions, shown without the model's answer and answered with a picker
- **Reviewer check:** about 5% of cards are Oxford test images with known labels, so my own accuracy as a reviewer is measured and shown.
- **Storage:** predictions and `label_event` rows are append-only. A label event records who proposed it (Oxford, a Commons category, me, the VLM), the review action and when.
- **Getting into training:**
  - Only reviewed labels enter a new dataset version, through the Step 2 pipeline like any other source.
  - My own cats are grouped by identity and capped at 20 images per cat per version.
  - Unreviewed predictions never become training labels.
  - Audit, golden-set and reviewer-check images are never trained on.

Done when: a batch of 50+ real photos and clips goes in, gets reviewed in the swipe app, and the reviewed labels show up in the next dataset version's manifest with their origin recorded.

**Step 10. Evaluation module and dbt.**

- `make eval` computes metrics per test set in Python (macro-F1, per-class recall with intervals, top-3, ECE, risk-coverage, false breed claim rate, unknown rejection) and writes them to metric tables.
- dbt builds on top of those tables and the operational ones:
  - staging models
  - incremental models over predictions and label events
  - a snapshot of label status
  - marts for accuracy, confused pairs, failure slices, abstention, review throughput, label sources and dataset composition per version
- Model contracts on the marts, plus dbt unit tests.

Done when: `dbt build` is green locally and in CI on fixtures.

**Step 11. Registry and promotion.**

- A `model_registry` table records weights hash, dataset version, config and metrics.
- The golden set is frozen.
- The champion vs challenger script uses a paired bootstrap on the same images. A challenger is promoted only if:
  - it wins on the golden set and the unseen-source test
  - no breed's recall drops more than the set margin
  - the false breed claim rate doesn't get worse
- The Model Health page reads only from marts.

**Step 12. Fine-tuning ablations.**

- First, frozen ViT-S against ViT-S with the last blocks unfrozen (fp16), on the same splits and seed.
- Then ViT-B/14 with the same protocol.
- A CI regression gate checks that a tiny fixture model reproduces its stored metrics.

**Phase 2 gate:** after 300+ reviews, v2 beats v1 under the paired test, shown from dbt. Tag `v0.3.0` and make the repo public.

### Phase 3: VLM screening and pet profiles

**Step 13. VLM image screener.**

- Ollama with `qwen3-vl:4b`, local tags only. It runs as a worker job, never during live view, because of the 6 GB VRAM limit.
- The prompt is versioned, and the model tag and digest are stored. JSON output is validated with Pydantic: one retry, then `parse_failed`, which is counted.
- Tags: real photo or drawing or toy, whole animal visible, kitten, blurry, more than one animal.
- The tags gate Commons training candidates and feed failure slices.

Done when: agreement per tag against my reviews (Cohen's kappa) and the effect on training-set noise are reported.

**Step 14. Breed-naming experiment.** A one-off script where the VLM names the breed zero-shot on the golden and unseen-source sets, compared with the classifier. The result goes in the README whichever way it lands. It is not a pipeline stage.

**Step 15. Pet profiles.**

- Detections are matched to known pets with pgvector nearest-neighbour search, plus appearance groups within a clip.
- Unmatched animals become candidate pets.
- I can name, merge or split pets. Every change is an append-only `identity_event`, and the dbt snapshot keeps the history.
- The Pets page shows photos, first and last seen, and the consensus breed with how consistently the model agrees with itself.

**Phase 3 gate:** VLM tags are measured, and my pets group correctly, with re-ID precision and recall on my own labelled photos reported, including failures. Tag `v0.4.0`.

### Phase 4: Live and presentation

**Step 16. Live camera page.**

- The same per-frame pipeline on the webcam, with probabilities smoothed over the last few frames for each appearance group.
- The VLM is never loaded at the same time.
- Webcam FPS is measured.

**Step 17. Demo and docs.**

- `make up && make demo` from a fresh clone, with a seeded demo database.
- README with results, diagrams, ERD, dbt lineage and limitations.
- `MODEL_CARD.md`, `DATASET_CARD.md` and a generated attribution list.
- ADRs for the main decisions.

**Step 18.** A 45-second demo video ("my breed model knows when it doesn't know"), a write-up, and tag `v1.0.0`.

## Deferred

- **Error predictor.** I'll revisit it once there are 150+ reviewed errors to train on.
- **ONNX Runtime.** PyTorch serves in v1. ONNX may come back as a measured speed comparison.
- **Commons expansion for dogs.** Dogs use the 25 Oxford breeds for now.

## Open decisions

- **Licence:** decide before the repo goes public. Leaning MIT or Apache 2.0. Oxford-IIIT images stay CC BY-SA and are never in the repo. Commons images are never in the repo either; an attribution list is generated from the manifest.
- **Stanford Dogs:** excluded. Its ImageNet terms allow non-commercial use only.
- **False breed claim target:** 10% or lower to start, revisited after Step 4.
- **Python version:** pinned to 3.12 for the widest PyTorch, RF-DETR and dbt support.
