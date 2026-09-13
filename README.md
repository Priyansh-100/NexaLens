# NexaLens - LLM-Powered Business Analytics Platform

A hybrid RAG (Retrieval-Augmented Generation) system that combines structured SQL querying with unstructured document search, powered by local LLMs (Llama 3.1, Qwen2.5) via Ollama.

## Features

- **Hybrid RAG**: Query both SQL databases and documents in a single natural language question
- **Text-to-SQL**: Convert natural language to executable PostgreSQL with schema awareness
- **Document Ingestion**: Process PDFs, CSVs, Excel, Word, Markdown with automatic chunking & embedding
- **Financial Modeling**: DCF, NPV, IRR, Payback, Sensitivity Analysis with Excel export
- **Forecasting**: Time-series forecasting (Prophet, ARIMA, ETS) with backtesting
- **Scheduled Reports**: Cron-based report delivery via email/Slack webhooks (PDF, Excel, CSV, HTML)
- **Local LLMs**: Runs entirely on-premise with Ollama (no API keys needed)
- **Streamlit UI**: Business-friendly chat interface with visualizations
- **FastAPI Backend**: Production-ready API with auth, metrics, health checks
- **Security**: SQL AST validation, read-only enforcement, statement timeouts, identifier validation
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
- 16GB+ RAM recommended
- NVIDIA GPU optional (for faster LLM inference)

### 1. Clone & Configure
```bash
cd NexaLens
cp .env.example .env  # Edit POSTGRES_PASSWORD, SECRET_KEY
```

### 2. Start Stack (CPU mode - works on Mac/any hardware)
```bash
docker-compose --profile cpu up -d
```

### 3. Start Stack (GPU mode - requires NVIDIA GPU)
```bash
docker-compose --profile gpu up -d
```

### 4. Pull Models
```bash
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull nomic-embed-text
```

### 5. Initialize Database
```bash
docker-compose exec api python -m nexalens.models.session
# OR use CLI:
docker-compose exec api python scripts/cli.py init-database
```

### 6. Create Admin User
```bash
docker-compose exec api python scripts/cli.py create-admin
# Follow prompts for email, password, name
```

### 7. Access
- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Chroma DB**: http://localhost:8001
- **Ollama**: http://localhost:11434

## API Usage

### Authentication
```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@company.com", "password": "secure123", "name": "Admin", "role": "admin"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'username=admin@company.com&password=secure123'
```

### Data Sources

#### SQL Source
```bash
curl -X POST http://localhost:8000/data-sources \
  -H "Authorization: Bearer <token>" \
  -F 'name=analytics_db' \
  -F 'type=sql' \
  -F 'config={"host": "postgres", "database": "analytics", "user": "postgres", "password": "postgres"}'
```

#### Document Source
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

### Query Examples

| Question | Intent |
|----------|--------|
| "What was total revenue by month in 2023?" | SQL |
| "Show me the refund policy for enterprise customers" | Document |
| "Compare Q3 revenue to the forecast in the board deck" | Hybrid |
| "Why did churn increase last quarter?" | Hybrid |
| "Build a DCF model with 15% discount rate, 3% terminal growth" | Financial Model |
| "Calculate NPV of cash flows [-100, 30, 40, 50] at 10% discount" | Financial Model |
| "Forecast revenue for next 12 months" | Forecast |
| "Predict churn rate for next quarter using Prophet" | Forecast |
| "Email me monthly revenue report every Monday 9am" | Schedule Report |
| "Schedule weekly dashboard delivery to Slack" | Schedule Report |

### Analytics Endpoints

```bash
# Financial Modeling
curl -X POST http://localhost:8000/analytics/financial-model \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "dcf",
    "assumptions": {"discount_rate": 0.12, "terminal_growth": 0.03},
    "cash_flows": [100000, 110000, 121000, 133100, 146410],
    "output_format": "excel"
  }'

# Forecasting
curl -X POST http://localhost:8000/analytics/forecast \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "table_name": "orders",
    "metric_column": "amount",
    "date_column": "order_date",
    "periods": 12,
    "frequency": "M",
    "model_type": "prophet",
    "data_source_ids": ["<uuid>"]
  }'

# Scheduled Reports
curl -X POST http://localhost:8000/reports/schedules \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Weekly Revenue",
    "query": "What was revenue last week?",
    "data_source_ids": ["<uuid>"],
    "cron_expression": "0 9 * * MON",
    "recipients": ["admin@company.com"],
    "format": "pdf"
  }'
```

## Configuration

Key settings in `.env`:
- `LLM_MODEL`: Ollama model (llama3.1:8b, qwen2.5:7b, etc.)
- `CHUNK_SIZE`: Document chunk size for embeddings
- `SQL_MAX_ROWS`: Max rows returned from SQL
- `CORS_ORIGINS`: Allowed frontend origins
- `POSTGRES_PASSWORD`: **Must change for production**
- `SECRET_KEY`: **Must change for production**

### Docker Profiles
```bash
# CPU-only (default, works everywhere)
docker-compose --profile cpu up -d

# GPU (requires NVIDIA GPU + nvidia-container-toolkit)
docker-compose --profile gpu up -d
```

## Project Structure

```
NexaLens/
├── src/nexalens/
│   ├── api/           # FastAPI routes & app
│   ├── analytics/     # Financial modeling, forecasting, scheduling
│   ├── core/          # Config, logging, exceptions, security
│   ├── documents/     # Document processing & ingestion
│   ├── models/        # Pydantic & SQLAlchemy models
│   ├── rag/           # Hybrid RAG orchestrator
│   ├── services/      # LLM, embeddings, vector store
│   └── sql/           # Text-to-SQL, schema introspection, safety
├── ui/                # Streamlit frontend
├── docker/            # Dockerfiles
├── scripts/           # Init scripts, CLI
├── templates/         # Jinja2 report templates
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

# CLI Admin
python scripts/cli.py create-admin
python scripts/cli.py init-database
```

## Security Features

- **SQL Safety**: AST parsing with sqlglot, read-only enforcement (SELECT/WITH only), forbidden keyword blocking
- **Statement Timeout**: Configurable per-query timeout (default 30s)
- **Identifier Validation**: Regex validation + schema allowlist for tables/columns
- **Authentication**: JWT with access/refresh tokens, OAuth2PasswordBearer
- **Authorization**: Role-based access control (Admin/Analyst/Viewer)
- **SSRF Protection**: Webhook allowlist, private IP blocking
- **Secrets**: Production startup fails if default SECRET_KEY/POSTGRES_PASSWORD used

## Production Deployment

1. Use proper secrets management for `SECRET_KEY`, `DATABASE_URL`
2. Set `APP_ENV=production`
3. Use PostgreSQL with pgvector extension
4. Configure Ollama with GPU support
5. Set up reverse proxy (nginx) with TLS
6. Enable monitoring (Prometheus metrics at `/metrics`)
7. Run scheduler as separate worker (Celery/Redis beat)
8. Use encrypted datasource credentials

## License

MIT