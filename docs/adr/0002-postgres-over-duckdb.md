# 0002. PostgreSQL with pgvector, not DuckDB or SQLite

- **Status:** Accepted
- **Date:** 2026-11-22

## Context

PetLens stores uploads, detections, predictions, review answers, pet identities, evaluation results and a job queue, and it needs nearest-neighbour search over 384- and 768-dimension embeddings for the evidence panel and pet matching. The API, the worker, the Streamlit page and dbt all touch the same data at the same time.

DuckDB is excellent for analytics and would make the dbt side simple, but it allows one writer process at a time. SQLite has the same limit and no vector type without extensions. A separate vector store (FAISS files, Chroma) would mean keeping two stores consistent.

## Decision

Use **PostgreSQL 16 with pgvector** in Docker for everything: transactional tables, the append-only event tables, the job queue, embeddings (cosine distance) and the dbt marts (separate schemas).

## Consequences

- One store, one backup, one set of migrations (Alembic). Transactions cover "store the prediction and assign the pet" together.
- Several processes can write at once (API, worker, review page).
- Triggers make `label_event` and `identity_event` append-only in the database itself, not just in code.
- dbt runs on the same server with dbt-postgres; no export step.
- Costs: Docker is needed even for the demo, and CI runs a Postgres service container. Accepted, since `make up` hides it.
