.PHONY: help setup setup-obs setup-rag setup-adk dev test lint fmt check sample index index-status agent-minimal clues-db judge agents-demo agents-routing probe probe-compare docker-build docker-up docker-down docker-logs verify-docker clean

VENV := .venv
PY   := $(VENV)/bin/python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install dependencies (requires uv)
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "\n>> Created .env -- add your API keys to it.")

setup-obs:  ## Add the optional Langfuse tracing dependency
	uv pip install -e ".[dev,obs]"

setup-rag:  ## Add the optional ChromaDB retrieval dependency
	uv pip install -e ".[dev,rag]"

setup-adk:  ## Add the optional ADK multi-agent dependencies (Gemini, MCP, A2A)
	uv pip install -e ".[dev,adk]"

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

sample:  ## Generate a local dataset sample from your own full download (gitignored)
	$(PY) scripts/make_sample.py

index:  ## Build the clue vector index (no-op if there is no clue data)
	$(PY) scripts/build_index.py

index-status:  ## Report index state without building anything
	$(PY) scripts/build_index.py --status

agent-minimal:  ## Run the single ADK agent with one tool, logging Think/Act/Observe
	$(PY) -m agents.minimal_agent

clues-db:  ## Export the clue TSV to SQLite for the MCP server
	$(PY) scripts/export_clues_db.py

judge:  ## Run the A2A judge agent (leave running in its own terminal)
	$(VENV)/bin/uvicorn agents.judge_agent:app --port 8001

agents-demo:  ## Run the multi-agent system (needs GOOGLE_API_KEY; start `make judge` first)
	$(PY) -m agents.run_demo

agents-routing:  ## Routing only: no MCP subprocess, no second server
	$(PY) -m agents.run_demo --no-mcp --no-a2a

probe:  ## Measure output corruption on the current prompt (costs 1 API call per trial)
	$(PY) scripts/probe_answer_quality.py --trials $(or $(TRIALS),10)

probe-compare:  ## Re-run the 7/8 vs 0/12 measurement: control vs production prompt
	$(PY) scripts/probe_answer_quality.py --trials $(or $(TRIALS),8) --compare

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
