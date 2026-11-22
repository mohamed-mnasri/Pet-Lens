# PetLens

**Local ML system with VLM data screening and human-in-the-loop evaluation**

Python · PyTorch · PostgreSQL / pgvector · dbt · Ollama · FastAPI

> Status: **pre-alpha.** This README describes the intended design and is updated with real results as each phase lands. See [docs/SPEC.md](docs/SPEC.md) for the full design and [docs/PLAN.md](docs/PLAN.md) for the build plan.

---

PetLens is an end-to-end machine learning system that runs entirely on one laptop. It has five parts:

- a versioned, licence-tracked image data pipeline
- a calibrated classifier that abstains when it isn't sure
- a human-in-the-loop review loop that feeds retraining
- model promotion gated by statistical tests
- a local vision-language model used as a measured data-quality screener

The domain is pet breed recognition. The engineering patterns are the ones production teams use for document classification, defect inspection and content moderation: selective prediction, label lineage, and champion vs challenger evaluation.

## What it demonstrates

| Area | What's built |
| --- | --- |
| **Data engineering** | Reproducible dataset builds from Oxford-IIIT Pet and Wikimedia Commons. Every image carries its licence and attribution. Near-duplicates are clustered, splits are grouped so nothing leaks between train and test, and manifests are versioned. Also: an operational PostgreSQL schema with Alembic migrations, append-only label events, a Postgres-backed job queue, and dbt marts with incremental models, snapshots and contracts. |
| **ML engineering** | DINOv2 embeddings with linear and MLP heads, then partial fine-tuning on a 6 GB GPU. Temperature calibration, open-set rejection with kNN in pgvector, and decision thresholds tuned on real-world data. |
| **Evaluation** | Metrics reported separately for in-domain, unseen-source, domestic-cat and unknown-breed test sets, with per-class intervals, a false-claim rate as the headline safety metric, and a paired bootstrap before any model is promoted. |
| **Human-in-the-loop** | A swipe review app with blind audits and reviewer-accuracy checks. Only reviewed labels ever reach training. |
| **Local AI** | A vision-language model (`qwen3-vl:4b` on Ollama) screens images as structured, schema-validated output. Its tags are scored against human review, not trusted. |
| **Backend** | FastAPI with Pydantic contracts, input hardening, upload limits with compression, a dead-letter table, and a background worker. |

Everything runs on one machine: **no cloud services, no paid APIs, €0 to build and run.**

## What it does

PetLens takes an **uploaded photo or short video** (and later a **PC camera feed**), finds the cats and dogs in it, and suggests the breed each one *looks most like*, with calibrated confidence. It never claims to *identify* a breed. Most cats have no breed, so "domestic cat, no clear breed" is a trained answer, not a failure.

Around each prediction:

- **Evidence, not just a label.** Every answer shows the training images it looks most like, and close breed pairs are shown side by side.
- **Review feeds the data.** I swipe predictions right or wrong. Reviewed labels go into the next dataset version with their origin recorded.
- **Measured improvement.** A new model is promoted only if it beats the current one on sources it has never seen, without making more false breed claims on ordinary cats.
- **Pet profiles.** My own pets are recognised across uploads by appearance.

## Planned architecture

```
Photo / 10 s clip (Upload)                Webcam (Live, Phase 4)
        │                                         │
        ▼                                         ▼
FastAPI ingest: limits, signature check, 1024 px WebP ── rejects to dead-letter
        │                                 │
        │ single files                    │ batches ──► Postgres job queue ──► worker
        ▼                                 ▼
Detect (RF-DETR) ── people blurred ── DINOv2 embedding ── per-species head
        │                    kNN open-set score + evidence (pgvector)
        ▼
PostgreSQL 16 + pgvector ──┬── Swipe review ──► append-only label events
                           ├── Evaluation (Python) + dbt Core marts
                           ├── VLM screener (qwen3-vl:4b via Ollama, batch)
                           └── Registry + paired promotion gate
                                        │
                                        ▼
        Streamlit UI: Upload · Review · Pets · Model Health · Data · Live
```

## Planned stack

| Layer | Technology |
| --- | --- |
| Detection | RF-DETR (Apache 2.0 sizes) |
| Classification | DINOv2 ViT-S/14 and ViT-B/14, linear / MLP heads, partial fine-tuning, PyTorch (CUDA, fp16) |
| Image and video | Pillow (WebP), PyAV, ffprobe |
| Local VLM | Ollama with `qwen3-vl:4b` (local models only, never `-cloud` tags) |
| API | FastAPI, Pydantic v2 |
| Storage | PostgreSQL 16 + pgvector, SQLAlchemy, Alembic, Postgres-backed job queue |
| Transformation | dbt Core + dbt-postgres (incremental models, snapshots, contracts) |
| UI | Streamlit |
| Quality | pytest, Ruff, mypy, pre-commit, GitHub Actions |

## Data

- **Training:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/): 37 breeds (12 cat, 25 dog), CC BY-SA 4.0. Downloaded by script; images are **not** committed to this repo.
- **Expansion:** [Wikimedia Commons](https://commons.wikimedia.org/) cat breed and domestic-cat categories, fetched through the API. Only CC0, public domain, CC BY and CC BY-SA files are used, with author and licence recorded per file. Not committed either.
- **Real-world testing and review:** my own photos and videos, checked through the swipe review app.

The reasoning, numbers and sources are in [docs/DATASET.md](docs/DATASET.md).

## Roadmap

- [ ] **Phase 1: Core.** Data pipeline, classifier with domestic class and open-set check, detection and clip grouping, Postgres schema, API, job queue, Upload page
- [ ] **Phase 2: Data engineering and feedback loop.** Swipe review with blind audit, evaluation and dbt marts, model registry and promotion gate, fine-tuning
- [ ] **Phase 3: VLM and pets.** VLM image screener, breed-naming experiment, pet profiles
- [ ] **Phase 4: Live and presentation.** Live camera, one-command demo, model and dataset cards, ADRs, demo video

## Development setup

Windows 11 is the main dev machine; Linux works the same way. Requires [uv](https://docs.astral.sh/uv/), Docker and GNU Make.

```bash
git clone https://github.com/mohamed-mnasri/petlens.git
cd petlens
cp .env.example .env      # set DATA_DIR to where datasets and weights should live
make install              # uv sync
make db-up                # Postgres 16 + pgvector in Docker
make check                # ruff, mypy, pytest
```

The DB smoke test runs only when `DATABASE_URL` is set in the environment.

## Responsible use

Breed identification from a photo is inherently uncertain, even for experts. Most real pets are mixed breed, and the model covers only 12 pedigree cat breeds plus domestic cats, and 25 dog breeds. PetLens shows top-3 suggestions with calibrated probabilities, abstains when unsure, and should **not** be used for adoption, legal, veterinary or insurance decisions.

## Licence

To be decided before the repository is made public. Third-party models and datasets keep their own licences.
