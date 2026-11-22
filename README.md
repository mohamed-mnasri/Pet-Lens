# PetLens

**A fully local pet breed recognition system that knows when it doesn't know, and learns from a fast human review loop.**

> Status: **pre-alpha.** This README describes the intended design. Step 1 (repo skeleton) is in progress; see [docs/PLAN.md](docs/PLAN.md) for the build plan and [docs/SPEC.md](docs/SPEC.md) for the full design. Sections will be updated with real results as each phase lands.

---

## What it does

PetLens takes an **uploaded photo or short video** (and later a **PC camera feed**), finds the cats and dogs in it, and suggests the breed each one *looks most like*, with honest, calibrated confidence. It never claims to *identify* a breed. Most cats have no breed, so "domestic cat, no clear breed" is a trained answer, not a failure.

What makes it more than a typical image classifier is what happens around the prediction:

- **Evidence, not just a label.** Every answer shows the training images it looks most like, and close breed pairs are shown side by side.
- **A swipe review loop.** I swipe predictions right or wrong. Only reviewed labels feed the next dataset version, and every label keeps its origin.
- **Measured improvement.** A new model is only promoted if a paired test shows it beats the current one on data from sources it has never seen, without claiming more breeds for ordinary cats.
- **A local vision-language model with a narrow, measured job.** It screens images (real photo or drawing, whole animal visible, kitten, blurry), and its tags are scored against my reviews.
- **Pet profiles.** PetLens recognises my own pets across uploads by appearance.

Everything runs on one laptop. **No cloud services, no paid APIs, €0 to build and run.**

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
