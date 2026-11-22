# 0001. Local-first layout on a Windows laptop

- **Status:** Accepted
- **Date:** 2026-11-22

## Context

PetLens has to run entirely on one Windows 11 laptop (RTX 2060, 6 GB VRAM, USB webcam) with no cloud services. It needs the webcam, the GPU, a Postgres database and, later, Ollama for the VLM image screener. CI runs on GitHub's Linux runners.

Docker Desktop can't pass a USB webcam through to a container without extra tooling, and GPU access from containers on Windows adds setup friction for little benefit on a single machine.

## Decision

- The Python app (API, Streamlit UI, camera capture, training and inference) runs **natively on Windows** in a `uv`-managed virtual environment.
- **Ollama runs natively** on Windows (from Step 12) and listens on localhost only.
- **Only Postgres (with pgvector) runs in Docker**, via `docker-compose.yml`, on a named volume. Docker Desktop's disk image sits on D:.
- **Datasets, model weights and footage live in `D:\petlens-data`**, configured through `DATA_DIR`. None of it is committed.
- **CI runs on Linux** with a Postgres service container and never downloads datasets or weights.
- `.gitattributes` forces LF line endings so Windows and Linux see identical files.

## Consequences

- Webcam and GPU access are simple and fast.
- A fresh setup needs a few native installs (uv, Docker Desktop, Make) instead of one `docker compose up`. Phase 4 adds a one-command `make up && make demo` to close that gap.
- Paths must stay OS-neutral in code (`pathlib`, settings from env), since the same code runs on Windows locally and Linux in CI.
