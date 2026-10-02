.PHONY: help install dev lint format typecheck security test test-unit test-integration run docker-build docker-up docker-down clean

PYTHON ?= python3
UV ?= $(shell which uv 2>/dev/null || echo "uv")

help:
	@echo "AegisLLM - Production-Grade Resilient LLM Gateway"
	@echo "================================================="
	@echo "Available commands:"
	@echo "  make install           Install production dependencies"
	@echo "  make dev               Install all dependencies including development tools"
	@echo "  make lint              Run ruff linter checks"
	@echo "  make format            Run ruff code formatter and fixers"
	@echo "  make typecheck         Run strict mypy type checking"
	@echo "  make security          Run bandit static security audit"
	@echo "  make test              Run all tests with pytest"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make run               Start the Gateway development server"
	@echo "  make docker-build      Build the Gateway Docker image"
	@echo "  make docker-up         Start the Gateway stack with Docker Compose"
	@echo "  make docker-down       Stop the Docker Compose services"
	@echo "  make clean             Remove cache directories and temporary files"

install:
	$(UV) pip install -e .

dev:
	$(UV) pip install -e ".[dev]"

lint:
	$(UV) run ruff check src tests docs

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy src tests

security:
	$(UV) run bandit -r src/ -ll

test:
	$(UV) run pytest

test-unit:
	$(UV) run pytest tests/unit

test-integration:
	$(UV) run pytest tests/integration

run:
	$(UV) run uvicorn presentation.main:app --host 0.0.0.0 --port 7860 --reload

docker-build:
	docker build -t aegisllm:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
