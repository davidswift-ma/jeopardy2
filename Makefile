.PHONY: help setup dev test lint fmt check sample docker-build docker-up docker-down docker-logs verify-docker clean

VENV := .venv
PY   := $(VENV)/bin/python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install dependencies (requires uv)
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "\n>> Created .env -- add your API keys to it.")

dev:  ## Run the app locally with auto-reload at http://localhost:8000
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Check formatting and lint rules
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

fmt:  ## Auto-format and auto-fix
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

check: lint test  ## Lint and test

sample:  ## Regenerate the committed dataset sample from the full download
	$(PY) scripts/make_sample.py

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start the container at http://localhost:8000
	docker compose up --build

docker-down:  ## Stop and remove the container
	docker compose down

docker-logs:  ## Tail container logs
	docker compose logs -f

verify-docker:  ## Full deployment check: build, both CPU archs, no-.env case, live answer
	./scripts/verify_docker.sh

clean:  ## Remove caches and the venv
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__ *.egg-info
