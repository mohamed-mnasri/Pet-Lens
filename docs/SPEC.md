# PetLens: project specification

The consolidated design for PetLens after the final design review. [PLAN.md](PLAN.md) has the order of work, and [DATASET.md](DATASET.md) has the data details. Where those two files and this one disagree, this file wins, and I fix the other two.

**Last updated:** 2026-11-22
**Status:** approved for implementation. Step 1 (repo skeleton) is in progress.

## 1. Objective

**PetLens is a local ML system with VLM data screening and human-in-the-loop evaluation.** It's built end to end on one laptop:

- a versioned image data pipeline
- a calibrated classifier that abstains when unsure
- a review loop that feeds retraining
- a statistically gated model promotion
- a local vision-language model used as a measured data-quality screener

The domain is pet breed recognition. Given a photo or short clip of a cat or dog, it says which breed the animal **looks most like**, or says plainly that it doesn't look like any breed.

The engineering story leads, and the pets make it easy to show. The story is the ML feedback loop: predict, review, version the data, retrain, evaluate, promote. Real teams run the same loop for document classification, defect inspection and content moderation.

**How it's presented.** The README title, repo description and CV line lead with the engineering, not the cats: "PetLens: Local ML system with VLM data screening and human-in-the-loop evaluation | Python, PyTorch, PostgreSQL/pgvector, dbt, Ollama". A recruiter skimming for six seconds should read AI and data engineering, not a tutorial classifier.

What I want the project to prove, next to [EuroGrid](https://github.com/mohamed-mnasri/eurogrid-intelligence):

- I can turn unstructured data (images, video) into clean, versioned, structured data.
- I can train, calibrate and evaluate my own model honestly, including on data it has never seen.
- I can build the data engineering around a model: an operational database, a job queue, label lineage, dataset and model versions.
- I can use a local LLM where it is genuinely good at the job, and measure it.

## 2. Users and use cases

| User | Use case |
| --- | --- |
| A pet owner | Upload a photo or a 10-second clip and get an honest answer: "looks like a Ragdoll, 87%", "domestic cat, no clear breed", or "not sure". |
| Me as the reviewer | Swipe through predictions as right or wrong, so the model improves without manual labelling. |
| Me as the ML engineer | See where the model fails (per breed, per source, per image condition) and decide whether a new model is better. |
| An interviewer or reader | Clone the repo, run one command, and see the whole loop working on a seeded demo database. |

It is not for adoption, veterinary, legal or insurance decisions. Breed from a photo is uncertain even for experts, and most pets have no breed.

## 3. Core features

1. **Upload and predict.** Photos and short clips go through detection, cropping, embedding, classification and the decision policy. The result card shows the top 3, the decision, and the five most similar training images for the suggested breed.
2. **Honest decisions.** There are three outcomes: *accept* ("looks like X"), *domestic / no clear breed*, and *not sure* (sent to review). Close breed pairs show both names. The wording is always "looks like", never "is".
3. **Domestic cat as a real class.** Most cats have no breed, so "domestic cat" is a trained class with its own test set, not just a fallback.
4. **Swipe review app.** Each card shows a crop, the suggested answer and reference photos. I swipe right (correct), left (wrong) or up (not sure), with keyboard shortcuts. Only reviewed labels go into training.
5. **Data pipeline.** A reproducible `make data` builds a versioned manifest from Oxford-IIIT Pet and Wikimedia Commons, with licence and attribution per image, near-duplicate clustering and grouped splits.
6. **Evaluation and promotion.** Metrics are reported per test set, with intervals. A challenger is promoted only if a paired test shows it really beats the champion.
7. **Job queue.** Batch reviews, Commons screening and VLM screening run as background jobs in Postgres with a worker process. The UI shows their progress.
8. **VLM image screener.** A local vision model (Ollama) tags images as real photo or drawing, whole animal visible, kitten, blurry, or several animals. The tags gate training data and drive failure analysis.
9. **Pet profiles.** The app recognises my own pets across uploads by appearance, and keeps a profile per pet.
10. **Live camera.** A webcam view with boxes and smoothed predictions, added in the last phase.

## 4. Deferred and removed

| Feature | Status | Why |
| --- | --- | --- |
| Error predictor (scikit-learn model of "is this prediction wrong?") | Deferred | v1 will have too few reviewed errors to train on. I'll revisit it at 150+ reviewed errors. |
| ONNX export and ONNX Runtime | Deferred | It's a second CUDA runtime on Windows for little gain. PyTorch serves in v1. It may come back later as a speed comparison. |
| VLM as an independent breed auditor in the pipeline | Replaced | Small VLMs are weak at fine-grained breeds, so it would flood review with false alarms. It now screens images (feature 8), and breed naming is a one-off experiment. |
| LLM-written profile cards | Removed | They repeat EuroGrid's grounded-text checks with less purpose. |
| ByteTrack and track-level voting | Removed | Video is sampled at 1 fps. Animals are grouped across frames by appearance instead. |
| Commons expansion for dogs | Deferred | Cats are the focus. Dogs use the 25 Oxford breeds with the same pipeline. |
| Stanford Dogs | Excluded | ImageNet terms allow non-commercial use only. |

## 5. Architecture

```
Photo / 10 s clip (Upload page)          Webcam (Live page, Phase 4)
        │                                         │
        ▼                                         ▼
FastAPI ingest: signature check, limits, EXIF fix + GPS strip,
1024 px WebP, originals deleted ──► dead-letter table on rejection
        │
        ├── small jobs: inline           ├── batches: job table ──► worker
        ▼                                ▼
Detect (RF-DETR) ── people blurred ── crop ── DINOv2 embedding
        │
        ▼
Per-species head + kNN open-set score (pgvector) ── calibrated decision policy
        │
        ▼
PostgreSQL 16 + pgvector (Alembic): predictions, crops, embeddings,
label events, jobs, datasets, models, pets
        │
        ├──► Swipe review app ──► label events (append-only)
        ├──► Python evaluation module ──► metric tables
        ├──► dbt Core marts (incremental, snapshots, contracts, unit tests)
        ├──► VLM screener (Ollama, batch only)
        └──► Registry + promotion gate ──► champion model
                                │
                                ▼
          Streamlit: Upload · Review · Pets · Model Health · Data · Live
```

**Boundaries:**

- The Streamlit UI only talks to the API. It never opens the database.
- The API binds to 127.0.0.1. There's no authentication, because this is a single-user local tool.
- The VLM and the live camera never run at the same time, because of the 6 GB of VRAM. The VLM only runs as a batch job.
- Label events and predictions are append-only. History is never overwritten.
- The model's own unreviewed predictions never become training labels.

## 6. Technology stack

| Layer | Choice | Why this and not something else |
| --- | --- | --- |
| Language and tooling | Python 3.12, uv, Ruff, mypy (strict), pytest, pre-commit | Already in place. 3.12 has the widest PyTorch, RF-DETR and dbt support. |
| Detection | RF-DETR, Apache 2.0 sizes only | Gives species and boxes for cats, dogs and people. |
| Embeddings and classifier | DINOv2 ViT-S/14 (then ViT-B/14), PyTorch with CUDA, fp16 | Strong frozen features for small data. Fits a 6 GB GPU. Apache 2.0 weights. |
| Classifier heads | scikit-learn logistic regression, then a small PyTorch MLP | Simple baselines picked on validation macro-F1. |
| Image and video I/O | Pillow (WebP), PyAV with `ffprobe` for duration checks | Duration is checked before decoding. Video is sampled at one frame per second. |
| Local LLM / VLM | Ollama with `qwen3-vl:4b`, local tags only | 3.3 GB, fits when nothing else is on the GPU. Apache 2.0. |
| API | FastAPI, Pydantic v2 | Typed contracts that mirror the database. |
| Database | PostgreSQL 16 + pgvector, SQLAlchemy 2, Alembic | Concurrent writes from the API and the worker, vector search for open-set scores, evidence and pets. It's new compared with EuroGrid's DuckDB. |
| Job queue | A Postgres `job` table with `FOR UPDATE SKIP LOCKED` and one worker | No extra infrastructure. It closes the "long requests block" gap EuroGrid documents. |
| Transformation | dbt Core + dbt-postgres | Incremental models, snapshots, contracts and unit tests, which are features EuroGrid didn't use. |
| Evaluation maths | Python module (NumPy, scikit-learn) writing to metric tables | ECE, reliability bins and bootstrap intervals are easier to write and test in Python than in SQL. |
| UI | Streamlit | Fast to build, and different from EuroGrid's React. |
| Infra | Docker Compose for Postgres only. Everything else is native on Windows. | The webcam and GPU are simpler natively. See ADR 0001. |
| CI | GitHub Actions, Linux, Postgres service container | No datasets or weights are downloaded in CI. Tiny fixture models are used instead. |

I considered and left out Airflow or Dagster (one machine, and the job table plus make targets is enough), MLflow (the registry table is enough), Redis or Celery (Postgres does the queue), and LangGraph or multi-agent setups (no task here needs an agent).

## 7. Dataset strategy

See [DATASET.md](DATASET.md) for numbers and sources. The decisions are:

- **Oxford-IIIT Pet is the core:** 12 cat and 25 dog breeds, CC BY-SA 4.0. I dedupe it myself (Visual Layer found 75 duplicates and 23 train/test leaks).
- **Wikimedia Commons is the expansion**, for cats only in v1. It's fetched through the MediaWiki API with a proper User-Agent, throttling and caching. Only CC0, public domain, CC BY and CC BY-SA files are kept, with author, licence and URL stored per file.
- **Domestic cat class.** It's built from the Commons domestic short-haired and long-haired categories, which have about 9,245 short-haired files (an upper bound).
- **Source balance.** Every pedigree breed also gets Commons images. Otherwise the model could learn "Oxford-style photo means pedigree, Commons-style photo means domestic" instead of learning the cats.
- **Unknown-breed set.** Commons breeds that Oxford doesn't have (Norwegian Forest Cat, Scottish Fold, Devon Rex, Exotic Shorthair), plus non-pet images. All 37 Oxford breeds stay in training.
- **Screening Commons.** The v0 model admits a Commons image to the training pool automatically only when its top-1 prediction matches the category. Everything else, and a random 5% of the matches, goes to the swipe review queue. VLM tags (Phase 3) then remove drawings, toys, heavy occlusion and multi-animal images.
- **Duplicates.** SHA-256 catches exact copies, and pHash plus DINOv2 cosine similarity catches near-duplicates, grouped into clusters.
- **Splits.** Splits are grouped by cat identity, then clip, then near-duplicate cluster, then Commons uploader. They're stratified with a fixed seed, and an image never changes split once assigned.
- **Test sets.** Each is reported separately:
  - in-domain (the Oxford test set)
  - unseen-source (held-out Commons uploaders plus my own photos)
  - domestic-cat
  - unknown-breed
  - golden (small, frozen)
- **Real-world calibration slice.** Held-out Commons uploaders and my own footage, kept apart from the tests, are used to tune the abstention thresholds.
- **Augmentation.** Random resized crop, flip, small rotation, mild brightness and contrast changes, webcam-style downscaling and JPEG compression, random erasing. No hue shifts or greyscale, because coat colour identifies breeds. Augmented copies never count toward image targets.
- **Expansion rule.** A new breed needs at least 150 checked training images from 3+ sources, plus 50 test images from a source never trained on.
- **Versioning.** A dataset version is the date plus a short hash of the manifest. Manifests, `breeds.yaml` and `sources.yaml` are committed. Images never are.

## 8. Model strategy

1. **Detection.** RF-DETR finds cats, dogs and people. Species comes from the detector. People are blurred in anything stored.
2. **Crops.** Detector-style crops with a 10% margin, padded square, at 224 px.
3. **Embedding.** DINOv2 ViT-S/14 frozen as the baseline. Embeddings are stored in Parquet for training and in pgvector for serving.
4. **Heads.** One per species:
   - cats: 12 Oxford breeds plus `domestic_cat`
   - dogs: 25 Oxford breeds

   A cat can never get a dog breed.
5. **Calibration.** Temperature scaling on the calibration split.
6. **Open-set score.** kNN distance from the crop's embedding to the training embeddings, using pgvector.
7. **Decision policy (config, not code):**
   - Accept when calibrated confidence and the open-set score both pass their thresholds.
   - Answer domestic when `domestic_cat` wins, or when a cat fails the breed thresholds but is clearly a cat.
   - Send to review when the result is unsure.
   - Thresholds are tuned on the real-world calibration slice, not in-domain.
8. **Clips.** One frame per second, at most 10 frames.
   - Detections across frames are grouped by appearance: DINOv2 embedding similarity plus a box-size sanity check.
   - Each group gets its own averaged answer, so a two-cat clip gives two answers.
   - Frames without an animal are skipped.
   - If the grouping is ambiguous, the clip returns per-frame results and says why.
9. **Live view (Phase 4).** The same per-frame pipeline. Probabilities are smoothed over the last few frames for each appearance group.
10. **Fine-tuning.** The first ablation is frozen ViT-S against ViT-S with the last blocks unfrozen (fp16), on the same splits and seed. ViT-B/14 follows with the same protocol. The winner is chosen by the promotion gate, not by in-domain accuracy.
11. **Serving.** PyTorch on CUDA, with model weights loaded once per process. The registry records the weights hash, dataset version, config and metrics for every model.

## 9. Database and data engineering

**Operational schema** (Alembic migrations, Pydantic contracts mirroring each table):

| Table | Holds |
| --- | --- |
| `upload` | One row per photo or clip: type, size, duration, status, timestamps |
| `frame` | Stored 1024 px WebP per photo or sampled frame, with hash and dimensions |
| `detection` | Box, species, detector score, crop path, embedding (`vector(384)` or `vector(768)`) |
| `prediction` | Model version, top 3 with calibrated probabilities, open-set score, decision, appearance group |
| `label_event` | Append-only: proposed label, origin (Oxford, Commons category, me, VLM), review action, reviewer time |
| `vlm_tag` | Screener tags per image, prompt version, model tag and digest |
| `job` | Type, payload, state, attempts, error, timestamps |
| `dataset_version` | Manifest hash, counts, created at |
| `model_registry` | Weights hash, dataset version, config, metrics, champion flag |
| `pet`, `pet_link`, `identity_event` | Pets, links from detections to pets, and append-only name, merge and split events |
| `dead_letter` | Rejected payloads with the reason |

**Pipelines:**

- `make data`: fetch, extract, ingest, normalise labels, detect and crop, quality checks, embed, dedupe, out-of-fold label audit, split, report. Every stage can be rerun safely and only does new work.
- `make review-batch`: queues a folder of my photos and clips as a job, and the worker predicts and queues review cards.
- `make eval`: the Python evaluation module computes metrics per test set and writes them to metric tables.
- `dbt build`:
  - staging models over the operational tables
  - incremental models over predictions and label events
  - a snapshot of label status
  - marts for accuracy, confused pairs, failure slices (by breed, source and VLM tag), abstention, review throughput, label sources, and dataset composition per version

**Data quality:**

- Pydantic validation at the API.
- dbt tests and model contracts on the marts.
- Great-expectations-style checks written as plain pytest on manifests: every row has a licence and attribution, and no group crosses splits.
- A quality report per dataset version.

## 10. AI functionality

The local VLM (`qwen3-vl:4b` through Ollama) has two jobs. Both are measured.

1. **Image screener (pipeline stage).** For each Commons or batch image, it answers in JSON validated by Pydantic:
   - real photo, drawing or toy
   - whole animal visible
   - kitten
   - blurry
   - more than one animal

   The prompt is versioned, and the model tag and digest are stored. Results gate training data and feed failure slices. Measurement: agreement per tag against my swipe reviews on a sample (Cohen's kappa), plus the effect on training-set noise.
2. **Breed-naming experiment (one-off script).** On the golden and unseen-source sets, the VLM names the breed zero-shot, and I compare it with the classifier. The result goes in the README whichever way it lands.

**Guardrails:**

- Structured output only, with one retry, then `parse_failed`, which is counted.
- The VLM never changes a label. It can only filter candidates or add tags.
- Batch only, never during live view.

**No agent.** Every AI task here is a fixed pipeline step, so plain Python is more reliable. EuroGrid already shows a guarded agent. PetLens shows the other half: deciding with data where a model belongs.

## 11. Human-in-the-loop review

- **Swipe app** (Streamlit page backed by the API):
  - Each card shows the crop, the suggested answer and 3 to 5 reference photos of that breed.
  - Right means correct, left means wrong, up means not sure. Arrow keys work too.
  - On "wrong", a quick picker offers the other top-3 answers, "domestic", "not a cat or dog" or "skip".
- **Questions I can answer reliably come first:**
  - real photo or not
  - whole animal visible
  - cat or not
  - domestic or clearly pedigree
  - which of my pets

  Fine breed calls are only confirmed when I actually know.
- **Queues:**
  - Commons disagreements and the 5% sample of agreements
  - my batches
  - the 5% blind audit of confident predictions, shown without the model's answer, with a picker
- **Reviewer check.** About 5% of cards are Oxford test images with known labels, so my own accuracy as a reviewer is measured and shown.
- **Rules:**
  - Only reviewed labels enter a new dataset version, through the normal pipeline.
  - Audit, golden and reviewer-check images are never trained on.
  - My own cats are capped at 20 images per cat per dataset version, grouped by identity.

## 12. Testing and evaluation

**Tests in CI:**

- Unit tests:
  - decision policy (every branch)
  - compression caps (a 40 MP photo stored under 250 KB)
  - each input limit with an over-limit file
  - signature checks
  - clip grouping
- Data tests: no group crosses splits, every manifest row has a licence, `make data` is idempotent on a fixture.
- API tests against a Postgres service container.
- Alembic upgrade and downgrade on an empty database.
- Job queue: claim, retry, crash recovery, no double claims.
- End-to-end smoke test: a fixture image through detect, embed, classify, decide and store, with tiny fixture models.
- Model behaviour tests: the prediction stays stable under a horizontal flip and mild JPEG compression.
- dbt build and unit tests on fixtures.
- Regression gate: a tiny fixture model must reproduce its stored metrics.

**Metrics,** reported per test set and never pooled:

- macro-F1
- per-class recall with 95% intervals
- top-3 accuracy
- ECE before and after calibration
- risk-coverage curve
- **false breed claim rate on domestic cats**, the headline safety number. My target is 10% or lower on the domestic test set, and I'll revisit it after the Step 4 baseline.
- unknown-breed rejection rate
- learning curve at 25 / 50 / 100% of training data

**Promotion gate:**

- The challenger must win on the golden set and the unseen-source test under a paired bootstrap, with a gain above noise.
- No breed's recall can drop more than a set margin.
- The false breed claim rate must not get worse.

## 13. Local execution requirements

| Item | Requirement |
| --- | --- |
| Machine | Windows 11 laptop, i7-10750H, 32 GB RAM, RTX 2060 6 GB, 720p USB webcam |
| GPU memory | Detector + DINOv2 ViT-S fit together for inference. `qwen3-vl:4b` (3.3 GB) runs alone as a batch job. ViT-B fine-tuning uses fp16 and small batches. |
| Disk | Datasets, weights and footage in `D:\petlens-data`. At the upload caps, 10,000 stored images stay under 3 GB. |
| Processes | Postgres in Docker, plus native API, worker, Streamlit and Ollama |
| Expected speeds | To be measured and reported, not promised: photo latency, clip latency, webcam FPS, VLM seconds per image |
| Training | Plugged in to mains power, because laptop GPUs throttle on battery |

## 14. Zero-cost verification

| Component | Cost | Notes |
| --- | --- | --- |
| Oxford-IIIT Pet | €0 | CC BY-SA 4.0, downloaded by script |
| Wikimedia Commons | €0 | Public API, no key, per-file free licences |
| DINOv2 weights | €0 | Apache 2.0 |
| RF-DETR | €0 | Apache 2.0 sizes only |
| Qwen3-VL via Ollama | €0 | Apache 2.0, local tags only, never `-cloud` |
| PostgreSQL, pgvector, dbt Core, FastAPI, Streamlit, PyTorch | €0 | Open source |
| Docker Desktop | €0 | Free for personal use |
| GitHub and GitHub Actions | €0 | Free plan minutes, no payment details |
| OBS Studio for the demo video | €0 | Open source |

No paid API, subscription, trial or cloud service is used anywhere.

## 15. Implementation phases

| Phase | Steps | Gate |
| --- | --- | --- |
| 1: Core | 1 setup · 2 dataset pipeline · 3 baseline classifier and screening · 4 open-set and decision policy · 5 detection and clip grouping · 6 schema, API, limits, hardening · 7 job queue and worker · 8 Upload page with evidence panel | A photo or clip in gives a calibrated decision and evidence out, every event is in Postgres, and CI is green. Tag `v0.2.0`. |
| 2: Data engineering and feedback loop | 9 swipe review app · 10 evaluation module and dbt · 11 registry and promotion gate · 12 fine-tuning ablations | After 300+ reviews, v2 beats v1 under the paired test, shown from dbt. Tag `v0.3.0` and make the repo public. |
| 3: VLM screening and pet profiles | 13 VLM screener · 14 breed-naming experiment · 15 pet profiles | VLM tags measured against my reviews. My pets are grouped correctly, with precision and recall reported. Tag `v0.4.0`. |
| 4: Live and presentation | 16 Live page · 17 one-command demo, README, cards, ADRs · 18 demo video and write-up | `make up && make demo` works from a fresh clone. Tag `v1.0.0`. |

## 16. GitHub deliverables

- README with results tables from the repo, architecture and data-flow diagrams, ERD, dbt lineage, screenshots and limitations.
- `docs/SPEC.md` (this file), `docs/PLAN.md`, `docs/DATASET.md`.
- `docs/MODEL_CARD.md` and `docs/DATASET_CARD.md`, partly generated from the registry and manifests.
- An attribution list generated from the manifest, covering every Commons file used.
- ADRs: local Windows layout, Postgres over DuckDB, job queue in Postgres, DINOv2 linear probe first, VLM as screener rather than auditor, appearance grouping instead of tracking.
- CI badge, tests, `make demo` with a seeded database, and no data or weights in git.

## 17. LinkedIn demonstration plan

Hook: **"My breed model knows when it doesn't know."**

The 45-second video, recorded at 1080p in OBS with captions:

1. **0–5 s:** the hook on screen.
2. **5–13 s:** upload a photo of my cat: "domestic cat, no clear breed", with the evidence panel.
3. **13–20 s:** upload a pedigree photo: "looks like a Ragdoll, 87%", with Birman shown as the close alternative.
4. **20–30 s:** the swipe review app, catching a confident mistake.
5. **30–38 s:** Model Health: v2 against v1 on the unseen-source test, and the false breed claim rate.
6. **38–45 s:** pet profiles and the end card: "100% local · €0 · open source" and the repo link.

Every number on screen comes from the repo. The post tells the development story: what the first model got wrong about ordinary cats, and how the data fixed it.

## 18. CV-relevant outcomes

- Built a reproducible image dataset pipeline (Oxford-IIIT and Wikimedia Commons) with per-file licence lineage, near-duplicate clustering and grouped, leak-free splits.
- Trained and calibrated a DINOv2-based breed classifier with open-set rejection, evaluated on unseen-source data with per-class intervals.
- Designed a human-in-the-loop review loop with blind audits and reviewer-accuracy checks, feeding versioned retraining and a statistically gated model promotion.
- Built a FastAPI and PostgreSQL / pgvector backend with Alembic migrations, a Postgres job queue and a worker, input hardening, and dbt marts with incremental models, snapshots and contracts.
- Used a local VLM (Ollama) as a measured data-quality screener, and reported where it does and doesn't help.

Each line only goes on the CV once it is true in the repo.

## 19. Known limitations and risks

| Severity | Risk | Mitigation |
| --- | --- | --- |
| High | Confident pedigree labels on ordinary cats | Domestic class, real-world threshold tuning, false breed claim rate as a gate metric |
| High | Commons label noise, especially in confusable pairs | Top-1 auto-admit only, swipe review of the rest, 5% noise sample, VLM screening, out-of-fold label audit |
| Medium | My breed judgements are unreliable | Reference photos on cards, "not sure" option, measured reviewer accuracy, reliable questions first |
| Medium | Source confound between Oxford and Commons | Commons images for every pedigree breed, per-source metrics |
| Medium | Re-ID fails on similar-looking pets (two black cats) | Reported honestly, with merge and split tools |
| Medium | 6 GB VRAM | VLM batch only, fp16, ViT-S by default |
| Low | Small per-breed test sets | Intervals on every per-class number, and no claims below 50 test images |

The model only covers 12 pedigree cat breeds plus domestic, and 25 dog breeds. Everything else should come out as domestic, unknown or not sure.

## 20. Readiness

**Ready for implementation.** The important decisions are made:

- data sources and licences
- the domestic class
- the unknown-breed set
- the review loop
- tracking replaced by appearance grouping
- the VLM's role
- the job queue
- the evaluation protocol

All mandatory requirements hold: €0, local inference, data quality first, professional presentation, and a scope I can build. Open items that don't block Step 1 or 2:

- the repo licence (before going public)
- the real usable Commons counts (Step 2 report)
- the false breed claim target (revisited after Step 4)
