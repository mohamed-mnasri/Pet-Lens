.PHONY: install lint format types test test-db check db-up db-down db-shell data data-oxford data-commons data-build install-ml embed train

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

train:
	uv run python -m petlens.models train
