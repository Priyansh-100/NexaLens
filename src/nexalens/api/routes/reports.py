from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from nexalens.api.auth import get_current_user
from nexalens.core.exceptions import ValidationError
from nexalens.core.logging import get_logger
from nexalens.core.rbac import require_analyst_or_admin, require_admin
from nexalens.models.database import ReportScheduleModel
from nexalens.models.schemas import (
    ReportExecution,
    ReportFormat,
    ReportSchedule,
    ReportScheduleCreate,
    ReportScheduleUpdate,
    User,
    UserRole,
)
from nexalens.models.session import get_db_session
from nexalens.analytics.scheduler import report_scheduler

router = APIRouter(prefix="/reports", tags=["reports"])
logger = get_logger(__name__)


@router.post("/schedules", response_model=ReportSchedule, status_code=status.HTTP_201_CREATED)
async def create_schedule(
    schedule_in: ReportScheduleCreate,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_analyst_or_admin),
) -> ReportSchedule:
    try:
        return await report_scheduler.create_schedule(schedule_in, session, current_user.id)
    except Exception as e:
        logger.error("create_schedule_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/schedules", response_model=list[ReportSchedule])
async def list_schedules(
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> list[ReportSchedule]:
    return await report_scheduler.get_schedules(session, current_user.id)


@router.get("/schedules/{schedule_id}", response_model=ReportSchedule)
async def get_schedule(
    schedule_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> ReportSchedule:
    schedule = await report_scheduler.get_schedule(schedule_id, session)
    if not schedule:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    if schedule.created_by != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return schedule


@router.patch("/schedules/{schedule_id}", response_model=ReportSchedule)
async def update_schedule(
    schedule_id: UUID,
    updates: ReportScheduleUpdate,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> ReportSchedule:
    schedule = await report_scheduler.get_schedule(schedule_id, session)
    if not schedule:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    if schedule.created_by != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    updated = await report_scheduler.update_schedule(schedule_id, updates, session)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    return updated


@router.delete("/schedules/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    schedule = await report_scheduler.get_schedule(schedule_id, session)
    if not schedule:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    if schedule.created_by != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    await report_scheduler.delete_schedule(schedule_id, session)


@router.post("/schedules/{schedule_id}/run", response_model=ReportExecution)
async def run_schedule_now(
    schedule_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> ReportExecution:
    schedule = await report_scheduler.get_schedule(schedule_id, session)
    if not schedule:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    if schedule.created_by != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    try:
        return await report_scheduler.run_now(schedule_id, session)
    except Exception as e:
        logger.error("manual_run_failed", schedule_id=str(schedule_id), error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/executions", response_model=list[ReportExecution])
async def list_executions(
    schedule_id: UUID | None = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> list[ReportExecution]:
    # Admins see all executions, regular users only see their own
    user_id = None if current_user.role == UserRole.ADMIN else current_user.id
    return await report_scheduler.get_executions(session, schedule_id, limit, user_id)