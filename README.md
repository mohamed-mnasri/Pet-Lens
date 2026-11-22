# PetLens

**A fully local pet breed recognition system with an AI auditor that checks its work, and pet profiles that learn each individual animal.**

> Status: **planning / pre-alpha.** This README describes the intended design. Nothing below is implemented yet; sections will be updated with real results as each phase lands.

---

## What it does

PetLens takes a **PC camera feed, an uploaded video or an uploaded picture**, finds the cats and dogs in it, and suggests their most likely breeds with honest, calibrated confidence. It never claims to *identify* a breed: it says what an animal *looks most like*, and answers "unknown or mixed breed" when it isn't sure.

What makes it different from a typical image classifier is what happens after the prediction:

- **AI auditor.** A second local AI model independently re-checks predictions. When the two disagree, a human decides. Dashboards score both AIs against human labels and answer one question with data: *does an AI auditor catch more mistakes than a simple confidence threshold?*
- **Pet profiles.** PetLens recognises the same animal across camera sessions and uploads, and builds a profile per pet: photos, sightings, consensus breed and how consistently the model agrees with itself.
- **Measured improvement.** Every confirmed label feeds evaluation and retraining. A new model is only promoted if it beats the current one on a blind, random audit sample.

Everything runs on one laptop. **No cloud services, no paid APIs, €0 to build and run.**

## Planned architecture

```
PC camera / video / photo
        │
        ▼
Detect + track (RF-DETR, ByteTrack) ── people blurred
        │
        ▼
Embed + classify (DINOv2 + fine-tuned head) ── calibrated, abstains on unknown breeds
        │
        ▼
FastAPI ingest (Pydantic contracts) ──► PostgreSQL + pgvector
                                              │
            ┌─────────────────┬───────────────┼───────────────────┐
            ▼                 ▼               ▼                   ▼
     AI auditor          Pet profiles     dbt Core marts      Retrain + promote
  (qwen3-vl:4b via      (re-ID, SCD2      (accuracy,          (golden set +
   Ollama, batch)        identities)       calibration,        blind audit)
                                           auditor scorecard)
                                              │
                                              ▼
                     Streamlit UI: Live · Upload · Review · Pets · Model Health
```

## Planned stack

| Layer | Technology |
| --- | --- |
| Detection + tracking | RF-DETR (Apache 2.0 sizes), `supervision` ByteTrack |
| Classification | DINOv2 ViT-S/14 → fine-tuned head, PyTorch (CUDA, fp16), ONNX Runtime |
| Local LLM / VLM | Ollama with `qwen3-vl:4b` (local models only, never `-cloud` tags) |
| API | FastAPI, Pydantic v2 |
| Storage | PostgreSQL 16 + pgvector, SQLAlchemy, Alembic |
| Transformation | dbt Core + dbt-postgres |
| UI | Streamlit |
| ML extras | scikit-learn (error predictor), temperature scaling |
| Quality | pytest, Ruff, mypy, pre-commit, GitHub Actions |

## Data

- **Training:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/): 37 breeds (12 cat, 25 dog), CC BY-SA 4.0. Downloaded by script; images are **not** committed to this repo.
- **Real-world testing:** the author's own photos and videos.

## Roadmap

- [ ] **Phase 1 — Core:** data pipeline, embeddings + classifier, detection + tracking, Postgres schema, FastAPI, Live page
- [ ] **Phase 2 — Data engineering:** review queue with blind audit, dbt evaluation marts, model registry + promotion gate, CI
- [ ] **Phase 3 — After dbt:** AI auditor (scorecard, error predictor, prompt evals) and pet profiles (re-ID, profile pages, fact-checked LLM profile cards)
- [ ] **Phase 4 — Presentation:** one-command setup, architecture docs, ADRs, demo video

## Responsible use

Breed identification from a photo is inherently uncertain, even for experts. Most real pets are mixed breed, and the training data covers only 37 pedigree breeds. PetLens shows top-3 suggestions with calibrated probabilities, abstains when unsure, and should **not** be used for adoption, legal, veterinary or insurance decisions.

## Licence

To be decided before the repository is made public. Third-party models and datasets keep their own licences.
