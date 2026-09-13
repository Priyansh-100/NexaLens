from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from nexalens.api.auth import get_current_user
from nexalens.core.exceptions import ValidationError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import QueryRequest, QueryResponse, User
from nexalens.models.session import get_db_session
from nexalens.rag.orchestrator import rag_orchestrator

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> QueryResponse:
    if not request.question.strip():
        raise ValidationError("Question cannot be empty")

    logger.info("query_received", user_id=str(current_user.id), question=request.question[:100])

    try:
        response = await rag_orchestrator.process_query(request, session, current_user.id)
        return response
    except Exception as e:
        logger.error("query_failed", error=str(e), user_id=str(current_user.id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/history")
async def query_history(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy import select, desc
    from nexalens.models.database import QueryLogModel

    result = await session.execute(
        select(QueryLogModel)
        .where(QueryLogModel.user_id == current_user.id)
        .order_by(desc(QueryLogModel.created_at))
        .limit(limit)
        .offset(offset)
    )
    logs = result.scalars().all()

    return [
        {
            "id": str(log.id),
            "question": log.question,
            "intent": log.intent,
            "answer": log.answer,
            "confidence": log.confidence,
            "processing_time_ms": log.processing_time_ms,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]