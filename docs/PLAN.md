# PetLens: build plan

Working roadmap for PetLens. I update this file as steps land. The design itself lives in [SPEC.md](SPEC.md) and the data choices in [DATASET.md](DATASET.md); this file is about order of work, scope per step and what "done" means.

**Current step:** 11. Model registry, golden set and promotion gate
**Last updated:** 2026-11-23

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
| 3 | Embeddings, baseline heads, calibration, Commons screening | 1: Core | Done |
| 4 | Open-set score and decision policy | 1: Core | Done (false breed claim rate 0.6% on 174 domestic test cats) |
| 5 | Detection, person blurring and clip grouping by appearance | 1: Core | Done (photo 0.1 s, 6-frame clip 0.7 s on an RTX 2060) |
| 6 | Database schema, contracts, API, input limits and hardening | 1: Core | Done (API and limit tests run in CI against Postgres) |
| 7 | Job queue and worker | 1: Core | Done (UI progress view comes with Step 8) |
| 8 | Upload page with evidence panel (Phase 1 gate) | 1: Core | Done (tag v0.2.0 after merging to main) |
| 9 | Swipe review app, blind audit and reviewer checks | 2: Data engineering | Done in code (waiting on my first 50+ reviews) |
| 10 | Evaluation module and dbt marts | 2: Data engineering | Done (`dbt build` runs in CI on fixtures) |
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
| 3b | DINOv2 near-duplicate leak check (held-out images with cosine 0.97 or more to a training image are left out of evaluation), Commons screening (top-1 agreement admits, disagreements and a seeded 5% sample wait for review; results committed to `data/screening/<version>.parquet`), retrain with the admitted photos, balanced class weights. `make screen`, `make train` | Done |

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

- RF-DETR (Apache-licensed sizes only) for cats, dogs and people. People's faces are blurred in anything stored.
- No tracker. Clips are sampled at one frame per second, and detections across frames are grouped by appearance (DINOv2 similarity plus a box-size check). Each group gets its own averaged answer, so a two-cat clip gives two answers.
- If grouping is ambiguous, the clip returns per-frame results and says why.

Done when: a test clip with one cat gives one answer, a test clip with two cats gives two, and photo and clip latency are measured.

Implementation notes:

- Code lives in `petlens.vision`: `detect` (RF-DETR wrapper, animal threshold 0.5, person threshold 0.3), `privacy` (Gaussian blur over the head region of each person box: the top of the box, about a seventh of a standing person or under a third of a close-up, so a pet held in someone's arms stays visible), `video` (PyAV, one frame per second, at most 10 frames, rotation metadata respected), `grouping` (greedy appearance grouping that never merges two detections from the same frame) and `pipeline` (photo and clip analysis).
- Breeds are classified on the original pixels; only the copy that would be stored is blurred.
- Grouping is called ambiguous when a detection is almost equally similar to two groups (within 0.05); the clip then returns per-frame results with the reason.
- `make analyse FILES="cat.jpg clip.mp4"` loads the latest run, the policy and the detector and prints each answer with its latency.
- RF-DETR scores cat and dog on the same box independently and the top label can flip between frames, so overlapping animal boxes (IoU 0.7 or more) are merged and the species is taken from whichever kNN index (cat or dog) the crop is closest to.
- Unit tests use a fake detector and fake embeddings, so CI needs neither torch nor the RF-DETR weights.

Results on my machine (RF-DETR base + DINOv2 on an RTX 2060):

- Models load in about 23 s. A photo takes 0.08 to 0.12 s after the first (about 1.2 s with CUDA warm-up).
- A 6-frame one-cat clip gives one group in 0.71 s (0.12 s per frame); a 6-frame two-cat clip gives two groups in 0.69 s.
- Known gaps: a cat held in someone's arms can be missed (the person box wins), faces are found from the person box rather than a face detector (a person lying down or upside down may not have the head at the top), and rotation metadata still needs a real phone clip to test.

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
- Video duration is read from the container metadata before any decoding (PyAV, which bundles FFmpeg, so there is no separate `ffprobe` to install on Windows).
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

Implementation notes:

- Step 6 lands in three parts: 6a schema, migrations and contracts; 6b the API; 6c limits and hardening.
- Tables: `upload`, `frame`, `detection` (with a `vector(384)` embedding), `prediction`, `label_event`, `job`, `dataset_version`, `model_registry` and `dead_letter`. `vlm_tag` arrives with Step 13 and the pet identity tables with the pet steps. `label_event` is append-only: a trigger refuses updates and deletes.
- Contracts (`petlens.contracts`) share their allowed values with the check constraints, and a test fails if a table and its contract drift apart. Another test compares the migrated database with the models.
- `make db-migrate` applies the migrations, `make api` serves the API on 127.0.0.1:8000 (interactive docs at `/docs`), and `make storage-report` shows disk use per folder. Limits live in `petlens.config` and can be overridden in `.env`.
- Prediction runs inside the request for now. Step 7 moves it onto the job queue, which is what `/jobs/{id}` will report on.
- Photos and sampled frames are resized to 1024 px before analysis, so what is analysed is what is stored. Stored files are named by the hash of the stored bytes; the uploader's file name is only kept, cleaned, as a label in the database.

**Step 7. Job queue and worker.**

- A `job` table claimed with `SELECT … FOR UPDATE SKIP LOCKED`, and one worker process (`make worker`).
- Jobs have states, attempts and an error column, and stuck jobs are recovered after a timeout.
- Batch reviews, Commons screening and later VLM screening all run here. Single photos and clips stay synchronous.

Done when: tests cover claim, retry, crash recovery and no double claims, and the UI shows job progress.

Implementation notes:

- `petlens.jobs.queue`: `enqueue`, `claim` (`FOR UPDATE SKIP LOCKED`, oldest runnable first), `heartbeat` (with progress and a partial result), `complete`, `fail` (retries after 30 s, 60 s, 120 s ... up to `max_attempts`, then `dead`) and `recover_stuck` (a running job with no heartbeat for 10 minutes is failed, so it retries or dies like any other failure). Times come from the database clock.
- The worker (`make worker`) runs one job at a time, sends a heartbeat every 30 s from a separate thread so a slow step doesn't look like a crash, and stops after the current job on Ctrl+C.
- Job kinds: `review_batch` (every photo and clip directly inside a folder, at most 200 files; each file is its own transaction, stored with upload kind `review`, and rejected files go to `dead_letter` with source `worker`; a retry skips files already stored) and `commons_screening` (the same as `make screen`).
- `make review-batch FOLDER=...` queues a folder, `make jobs-status` lists recent jobs with progress, and `POST /jobs/review-batch` does the same from the API. `GET /jobs/{id}` returns state, progress and result, which the Step 8 page will show.

**Step 8. Upload page.**

- Streamlit upload grid that shows the limits up front and rejects oversize files before uploading them.
- Result cards show the top 3 with calibrated probabilities, the decision badge, and the five most similar training images for the suggested breed (from pgvector).

**Phase 1 gate:** a photo or clip in gives a calibrated decision and evidence out, every event is in Postgres, an end-to-end smoke test with tiny fixture models runs in CI, and CI is green. Tag `v0.2.0`.

Implementation notes:

- 8a (backend): `train_image` holds the training pool's embeddings (`make db-index` copies them from the latest run; rerun it after `make train`). Each run returns, per prediction, the five nearest training photos of the suggested breed from pgvector (cosine distance); a clip group uses the mean embedding of its detections, which now record their group. `/images/{train,frames,crops}/{hash}` serve only content-hash names from known folders, and `GET /jobs` and `GET /limits` feed the page.
- 8b (page): `make ui` opens the Streamlit page on 127.0.0.1:8501. It shows the limits up front, checks type, size and megapixels before sending (and Streamlit's own upload cap is 25 MB), then shows each upload with boxes drawn on the stored, face-blurred frames, a decision badge, the top 3 as bars, the kNN distance and the five evidence photos with similarity, credit and licence. A second tab queues review batches and shows job progress. The page only talks to the API.
- The end-to-end smoke test sends a photo through the page's own client to the API with tiny fixture models (a colour detector, colour embeddings and a logistic head) and checks the decision, the evidence and the drawn frame. A Streamlit `AppTest` checks the page renders.

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

Implementation notes:

- 9a (cards): `review_card` (migration 0004) holds one card per photo to check, in four queues. `make review-load` queues the Commons cards from the latest screening (disagreements first when the model said domestic, then other disagreements, then the 5% sample) and reviewer-check cards (Oxford test photos, about 5% of all cards; half show the true breed and half a wrong one, so "correct" isn't always right). Review batches add a card per animal (per appearance group in clips), and a seeded 5% of confident answers become blind audit cards. `GET /review/next`, `POST /review/{id}` and `GET /review/stats` serve the page; each answer is an append-only `label_event` with origin `me`, the action (`accept`, `fix`, `reject` for "not a cat or dog", `unsure`, `skip`) and the time taken.
- 9b (page): the Review tab shows the photo with its box, the question, three buttons with arrow-key shortcuts (→ correct, ← wrong, ↑ not sure), a picker after "wrong" (other top answers, domestic, not a cat or dog, skip) and four training photos of the suggested breed. Blind audit cards show only the picker. The header shows what's left and my accuracy on check cards.
- 9c (into the data): `make review-export` writes my Commons reviews to `data/reviews/labels.parquet` (committed) and my reviewed batch animals as the `review` source (a crop of the face-blurred frame, licence `own-photo`, grouped per upload). `make data-build` then applies them: confirmed or corrected labels get `label_origin = me`, "not a cat or dog" is excluded, and "not sure" and blind-audit photos are never trained on. Check cards never change labels.
- 9d (weak photos): the first swipe session showed many photos that teach little: black and white, blurry, only paws or a nose, or the animal a speck. `make data-quality` checks every photo once (cached in `D:\petlens-data\processed\photo_checks.json`): colour spread, sharpness (Laplacian variance at 512 px), and a box from the detector for photos that have none (Commons). `make data-build` turns that into the flags `grayscale`, `blurry`, `cut_off` (box touches all four edges), `small_animal` (under 4% of the photo) and `no_animal`. Flagged photos keep their split so test sets stay stable, but they are never trained on and never queued for review, unless I already reviewed one as fine. `make review-load` takes unanswered weak cards off the queue. Batch cards use the frame where the animal is whole, and the upload page asks for a clear colour photo of the whole animal from the front, with a tip under a weak result.
- 9e (my own breed photos): the page's "Add training photos" tab takes a batch of photos of one breed (pick or type the breed, upload, then start the next batch). `POST /training-photos` checks each photo (one cat or dog of the right species, not weak), blurs faces, stores it and lists it in the `own` source with `label_origin = me` and licence `own-photo`; the same photo twice is kept once. `make own-photos` lists the counts per breed. `make data-build` splits them per photo like Commons photos. Ragdoll and Birman come first: they are a close pair, and every training report now has a look-alike table (per test set: how often each breed is taken for the other). `make compare` shows the table for the last two runs, for a before/after look; the Oxford in-domain test is the fair comparison because it doesn't change. I chose not to up-weight the pair: clean photos of both breeds are the real fix, and extra weight tends to make the model call those breeds too often. "Not sure" answers are now excluded completely (no training, no tests, no evidence).
- 9f (more breeds and photos): cats cover 21 breeds plus the domestic class, dogs 29 breeds (Oxford's 25 plus Golden Retriever, German Shepherd, French Bulldog and Poodle). New cat breeds: Scottish Fold, Norwegian Forest Cat, Himalayan, Devon Rex, Siberian, American Shorthair, Burmese, Cornish Rex, Oriental Shorthair and Savannah. I chose not to cover Bombay (an Oxford breed), Exotic Shorthair and Turkish Angora: they join Chartreux, Turkish Van, Korat, Selkirk Rex and Manx in the unknown-breed test, and an unknown breed now overrides a split inherited from an older version. `make import-folders FOLDER=...` imports one subfolder per breed into the `licensed` source; Pexels, Unsplash and Pixabay file names get their licence, others the given default, and every imported photo is flagged `no_display` (trains, never shown as evidence or examples). `make data-openverse` adds about 30 openly licensed photos per cat and dog breed from Openverse (CC0, public domain, CC BY, CC BY-SA; Wikimedia results skipped), and every Openverse photo gets a review card. My own and licensed photos are trusted like Oxford and seed screening for breeds Oxford doesn't have; Openverse photos are screened like Commons ones, and breeds with no trusted photos wait unused.
- 9g (review and look): "Not sure" deletes the photo (`make review-export` removes the stored copy and the build drops it); Skip changes nothing. After "Wrong" a box takes any breed of the card's species. The review tab filters cats or dogs. The page has a warm theme, card-style results with the evidence under "Why", and clearer tabs (UI polish was Step 17; done early because the plain page was hard to use).
- Not yet: the per-cat cap of 20 images waits for the pet identity steps, and the "which of my pets" question comes with them.

**Step 10. Evaluation module and dbt.**

- `make eval` computes metrics per test set in Python (macro-F1, per-class recall with intervals, top-3, ECE, risk-coverage, false breed claim rate, unknown rejection) and writes them to metric tables.
- dbt builds on top of those tables and the operational ones:
  - staging models
  - incremental models over predictions and label events
  - a snapshot of label status
  - marts for accuracy, confused pairs, failure slices, abstention, review throughput, label sources and dataset composition per version
- Model contracts on the marts, plus dbt unit tests.

Done when: `dbt build` is green locally and in CI on fixtures.

Implementation notes:

- 10a (evaluation): `make eval` loads the latest run's heads, its manifest, embeddings and `policy.yaml`, and writes one `eval_run` with `eval_metric` (top-1 with Wilson intervals, top-3, macro-F1, ECE, risk at 50/80/100% coverage, policy coverage, accuracy when accepted, no-clear-breed and review rates, false breed claim rate, unknown rejection rate), `eval_class_metric` (recall with Wilson intervals, precision) and `eval_image` (one row per test image with its prediction, decision and photo flags). Near-copies of training images are left out, as in training. The manifest's composition goes to `dataset_count` (replaced per version). Migration 0005.
- 10b (dbt): `dbt/` holds staging views over the operational and evaluation tables, incremental `int_predictions` (by time) and `int_label_events` (by the append-only id), a `snap_label_status` snapshot of review cards, and seven marts with enforced contracts: accuracy, confused pairs, failure slices (by source, label origin, breed and photo flag), abstention (test sets and live uploads per day), review throughput (with check-card accuracy), label sources and dataset composition. Two dbt unit tests cover confused pairs and review throughput. `make dbt` reads DATABASE_URL (so port 5433 works), and CI runs `dbt build` on fixtures from `python -m petlens.eval.fixtures`. dbt is its own dependency group.

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
