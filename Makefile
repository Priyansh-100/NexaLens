.PHONY: help install dev test lint format typecheck clean docker-up docker-down docker-logs

help:
	@echo "NexaLens - Development Commands"
	@echo ""
	@echo "  install       Install dependencies"
	@echo "  dev           Run API with hot reload"
	@echo "  ui            Run Streamlit UI"
	@echo "  test          Run tests"
	@echo "  lint          Run ruff linter"
	@echo "  format        Format code with ruff"
	@echo "  typecheck     Run mypy type checking"
	@echo "  clean         Clean cache files"
	@echo "  docker-up     Start Docker stack"
	@echo "  docker-down   Stop Docker stack"
	@echo "  docker-logs   View Docker logs"
	@echo "  eval          Run evaluation"

install:
	pip install -e ".[dev,ui]"

dev:
	uvicorn nexalens.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501

test:
	pytest -v --tb=short

test-cov:
	pytest --cov=nexalens --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down -v

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

init-models:
	docker-compose exec ollama ollama pull llama3.1:8b
	docker-compose exec ollama ollama pull nomic-embed-text

init-db:
	docker-compose exec api python -c "from nexalens.models.session import init_db; import asyncio; asyncio.run(init_db())"

eval:
	python -m nexalens.evaluation.runner