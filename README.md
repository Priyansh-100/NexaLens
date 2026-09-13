# NexaLens - LLM-Powered Business Analytics Platform

A hybrid RAG (Retrieval-Augmented Generation) system that combines structured SQL querying with unstructured document search, powered by local LLMs (Llama 3.1, Qwen2.5) via Ollama.

## Features

- **Hybrid RAG**: Query both SQL databases and documents in a single natural language question
- **Text-to-SQL**: Convert natural language to executable PostgreSQL with schema awareness
- **Document Ingestion**: Process PDFs, CSVs, Excel, Word, Markdown with automatic chunking & embedding
- **Local LLMs**: Runs entirely on-premise with Ollama (no API keys needed)
- **Streamlit UI**: Business-friendly chat interface with visualizations
- **FastAPI Backend**: Production-ready API with auth, metrics, health checks
- **Docker Compose**: One-command local development stack

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Streamlit │────▶│   FastAPI   │────▶│  PostgreSQL │
│     UI      │     │   Backend   │     │  (pgvector) │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼────┐  ┌────▼─────┐
              │  Ollama  │  │  Chroma  │
              │ (LLM/Emb)│  │ (Vectors)│
              └──────────┘  └──────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (optional, for faster LLM inference)
- 16GB+ RAM recommended

### 1. Clone & Configure
```bash
cd NexaLens
cp .env.example .env  # Edit as needed
```

### 2. Start Stack
```bash
docker-compose up -d
```

### 3. Pull Models
```bash
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull nomic-embed-text
```

### 4. Initialize Database
```bash
docker-compose exec api python -c "from nexalens.models.session import init_db; import asyncio; asyncio.run(init_db())"
```

### 5. Access
- **API**: http://localhost:8000/docs
- **UI**: http://localhost:8501
- **Chroma**: http://localhost:8001
- **Ollama**: http://localhost:11434

## Default Credentials
Create a user via API:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@company.com", "password": "secure123", "name": "Admin", "role": "admin"}'
```

## Adding Data Sources

### SQL Source
```bash
curl -X POST http://localhost:8000/data-sources \
  -H "Authorization: Bearer <token>" \
  -F 'name=analytics_db' \
  -F 'type=sql' \
  -F 'config={"host": "postgres", "database": "analytics", "user": "postgres", "password": "postgres"}'
```

### Document Source
```bash
curl -X POST http://localhost:8000/data-sources \
  -H "Authorization: Bearer <token>" \
  -F 'name=reports' \
  -F 'type=document' \
  -F 'config={}'

# Upload documents
curl -X POST http://localhost:8000/data-sources/<source_id>/documents \
  -H "Authorization: Bearer <token>" \
  -F 'file=@quarterly_report.pdf'
```

## Query Examples

| Question | Intent |
|----------|--------|
| "What was total revenue by month in 2023?" | SQL |
| "Show me the refund policy for enterprise customers" | Document |
| "Compare Q3 revenue to the forecast in the board deck" | Hybrid |
| "Why did churn increase last quarter?" | Hybrid |

## Configuration

Key settings in `.env`:
- `LLM_MODEL`: Ollama model (llama3.1:8b, qwen2.5:7b, etc.)
- `CHUNK_SIZE`: Document chunk size for embeddings
- `SQL_MAX_ROWS`: Max rows returned from SQL
- `CORS_ORIGINS`: Allowed frontend origins

## Project Structure

```
NexaLens/
├── src/nexalens/
│   ├── api/           # FastAPI routes & app
│   ├── core/          # Config, logging, exceptions
│   ├── documents/     # Document processing & ingestion
│   ├── models/        # Pydantic & SQLAlchemy models
│   ├── rag/           # Hybrid RAG orchestrator
│   ├── services/      # LLM, embeddings, vector store
│   └── sql/           # Text-to-SQL & schema introspection
├── ui/                # Streamlit frontend
├── docker/            # Dockerfiles
├── scripts/           # Init scripts
└── tests/             # Test suite
```

## Development

```bash
# Install deps
pip install -e ".[dev,ui]"

# Run API
uvicorn nexalens.api.main:app --reload

# Run UI
streamlit run ui/app.py

# Run tests
pytest

# Lint
ruff check .
mypy src/
```

## Production Deployment

1. Use proper secrets management for `SECRET_KEY`, `DATABASE_URL`
2. Set `APP_ENV=production`
3. Use PostgreSQL with pgvector extension
4. Configure Ollama with GPU support
5. Set up reverse proxy (nginx) with TLS
6. Enable monitoring (Prometheus metrics at `/metrics`)

## License

MIT