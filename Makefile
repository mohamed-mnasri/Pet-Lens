.PHONY: install lint format types test test-db check db-up db-down db-shell db-migrate db-downgrade api storage-report data data-oxford data-commons data-build install-ml embed screen train analyse

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

types:
	uv run mypy

test:
	uv run pytest

check: lint types test

db-up:
	docker compose up -d --wait db

db-down:
	docker compose down

db-shell:
	docker compose exec db psql -U petlens -d petlens

db-migrate:
	uv run python -m petlens.db upgrade

db-downgrade:
	uv run python -m petlens.db downgrade

# API on http://127.0.0.1:8000 (docs at /docs). Needs the database and the ml extra.
api:
	uv run --extra ml python -m petlens.api

storage-report:
	uv run python -m petlens.storage

# Dataset pipeline (Step 2). Safe to rerun: only new work is done.
data: data-oxford data-commons data-build

data-oxford:
	uv run python -m petlens.data oxford

data-commons:
	uv run python -m petlens.data commons

data-build:
	uv run python -m petlens.data build

# Step 3: embeddings and baseline heads. Needs the optional ML packages (torch, timm).
install-ml:
	uv sync --extra ml

embed:
	uv run --extra ml python -m petlens.models embed

screen:
	uv run python -m petlens.models screen

train:
	uv run python -m petlens.models train

# Step 5: analyse photos or clips, e.g. make analyse FILES="D:/petlens-data/footage/cat.mp4"
analyse:
	uv run --extra ml python -m petlens.vision $(FILES)
