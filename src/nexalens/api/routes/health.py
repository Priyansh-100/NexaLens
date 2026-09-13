from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.config import get_settings
from nexalens.core.logging import get_logger
from nexalens.models.schemas import HealthResponse
from nexalens.models.session import get_db_session
from nexalens.services.embeddings import embedding_service
from nexalens.services.llm import llm_service
from nexalens.services.vector_store import vector_store

router = APIRouter(tags=["health"])
logger = get_logger(__name__)
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_db_session)) -> HealthResponse:
    checks = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    try:
        checks["llm"] = await llm_service.health_check()
    except Exception:
        checks["llm"] = False

    try:
        checks["embeddings"] = await embedding_service.health_check()
    except Exception:
        checks["embeddings"] = False

    try:
        checks["vector_store"] = await vector_store.health_check()
    except Exception:
        checks["vector_store"] = False

    all_healthy = all(checks.values())
    any_healthy = any(checks.values())

    if all_healthy:
        status = "healthy"
    elif any_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version=settings.app_name,
        checks=checks,
    )


@router.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    return {"status": "ready"}