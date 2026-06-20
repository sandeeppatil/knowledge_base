.PHONY: help install dev-install test lint format type-check docker-up docker-dev clean

help:
	@echo "Knowledge Base Platform — Available targets"
	@echo "  install       Install production dependencies"
	@echo "  dev-install   Install development dependencies"
	@echo "  test          Run test suite with coverage"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run ruff formatter"
	@echo "  type-check    Run mypy type checker"
	@echo "  docker-up     Start production stack (API + Qdrant)"
	@echo "  docker-dev    Start development stack with hot-reload"
	@echo "  docker-down   Stop all containers"
	@echo "  clean         Remove build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

docker-up:
	docker compose --profile production up -d

docker-dev:
	docker compose --profile dev up

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f kb-api

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

# Create data directories
dirs:
	mkdir -p data/knowledge_bases data/models data/uploads data/logs data/faiss_indices
