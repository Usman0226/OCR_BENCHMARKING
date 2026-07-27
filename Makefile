# =============================================================================
# OCR Benchmark Framework — Makefile
# Usage: make <target>
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

DOCKER_COMPOSE := docker compose
EXEC          := $(DOCKER_COMPOSE) exec ocr
PYTHON        := python

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: build
build: ## Build Docker images (uses BuildKit cache)
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 $(DOCKER_COMPOSE) build

.PHONY: up
up: ## Start all services in the background
	$(DOCKER_COMPOSE) up -d

.PHONY: down
down: ## Stop and remove containers (keeps volumes)
	$(DOCKER_COMPOSE) down

.PHONY: down-volumes
down-volumes: ## Stop containers AND remove all named volumes (destructive!)
	$(DOCKER_COMPOSE) down -v

.PHONY: restart
restart: down up ## Restart all services

.PHONY: logs
logs: ## Follow logs from all services
	$(DOCKER_COMPOSE) logs -f

.PHONY: ps
ps: ## Show service status
	$(DOCKER_COMPOSE) ps

.PHONY: shell
shell: ## Open a bash shell in the ocr container
	$(EXEC) bash

# =============================================================================
# OCR pipeline (runs inside the container)
# =============================================================================

.PHONY: run-paddle
run-paddle: ## Run PaddleOCR on images/ → ocr_output/paddle/
	$(EXEC) $(PYTHON) run_ocr.py --engine paddle --images /workspace/images

.PHONY: run-tesseract
run-tesseract: ## Run Tesseract on images/ → ocr_output/tesseract/
	$(EXEC) $(PYTHON) run_ocr.py --engine tesseract --images /workspace/images

.PHONY: run-all
run-all: ## Run both engines on images/
	$(EXEC) $(PYTHON) run_ocr.py --engine all --images /workspace/images

.PHONY: score
score: ## Score OCR vs annotations → results.csv + results_summary.md
	$(EXEC) $(PYTHON) score.py

.PHONY: score-paddle
score-paddle: ## Score PaddleOCR only
	$(EXEC) $(PYTHON) score.py --engine paddle

.PHONY: score-tesseract
score-tesseract: ## Score Tesseract only
	$(EXEC) $(PYTHON) score.py --engine tesseract

.PHONY: visualize
visualize: ## Draw bounding boxes → reports/visualizations/
	$(EXEC) $(PYTHON) scripts/visualize.py

.PHONY: pipeline
pipeline: run-all score visualize ## Full pipeline: OCR → score → visualize

# =============================================================================
# Code quality (run inside container)
# =============================================================================

.PHONY: test
test: ## Run unit tests with coverage
	$(EXEC) $(PYTHON) -m pytest tests/ -v --tb=short --cov=ocr --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Run tests without coverage (faster)
	$(EXEC) $(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: lint
lint: ## Run ruff linter
	$(EXEC) $(PYTHON) -m ruff check ocr/ scripts/ run_ocr.py score.py

.PHONY: format
format: ## Format code with ruff
	$(EXEC) $(PYTHON) -m ruff format ocr/ scripts/ run_ocr.py score.py

.PHONY: typecheck
typecheck: ## Run mypy type checker
	$(EXEC) $(PYTHON) -m mypy ocr/ run_ocr.py score.py

.PHONY: check
check: lint typecheck test ## Run all quality checks

# =============================================================================
# Setup
# =============================================================================

.PHONY: setup
setup: ## First-time setup: copy .env + build + start
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example — edit it before continuing.")
	$(MAKE) build
	$(MAKE) up
	@echo ""
	@echo "All done. Label Studio is at http://localhost:8080"
	@echo "Run: make run-all && make score"

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-outputs
clean-outputs: ## Remove all generated OCR/scoring outputs (not source code)
	rm -f results.csv results_summary.md
	rm -rf ocr_output/paddle/* ocr_output/tesseract/*
	rm -rf reports/
	@echo "Cleared: ocr_output/*, results.csv, results_summary.md, reports/"
