from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from nexalens.api.routes import analytics, auth, data_sources, health, organizations, query, reports
from nexalens.analytics.scheduler import report_scheduler
from nexalens.core.config import get_settings
from nexalens.core.logging import get_logger, setup_logging
from nexalens.core.rate_limit import add_rate_limit_headers, rate_limiter
from nexalens.models.session import close_db, init_db

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("application_starting", env=settings.app_env)
    await init_db()
    report_scheduler.start()
    yield
    report_scheduler.shutdown()
    await rate_limiter.close()
    await close_db()
    logger.info("application_shutdown")


app = FastAPI(
    title="NexaLens",
    description="LLM-powered business analytics platform with hybrid RAG",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url="/redoc" if settings.app_env != "production" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limit headers middleware
app.middleware("http")(add_rate_limit_headers)

if settings.enable_metrics:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(query.router)
app.include_router(data_sources.router)
app.include_router(analytics.router)
app.include_router(reports.router)
app.include_router(organizations.router)


@app.get("/")
async def root():
    return {"name": "NexaLens", "version": "0.1.0", "status": "running"}