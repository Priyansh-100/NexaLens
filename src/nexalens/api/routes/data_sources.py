from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from nexalens.api.auth import get_current_user
from nexalens.core.exceptions import ValidationError
from nexalens.core.logging import get_logger
from nexalens.models.database import DataSourceModel
from nexalens.models.schemas import DataSource, DataSourceType, User
from nexalens.models.session import get_db_session
from nexalens.documents.service import document_service

router = APIRouter(prefix="/data-sources", tags=["data-sources"])
logger = get_logger(__name__)


@router.get("", response_model=list[DataSource])
async def list_data_sources(
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    result = await session.execute(select(DataSourceModel).where(DataSourceModel.is_active == True))
    sources = result.scalars().all()
    return [
        DataSource(
            id=s.id,
            name=s.name,
            type=s.type,
            config=s.config,
            description=s.description,
            created_at=s.created_at,
            updated_at=s.updated_at,
            is_active=s.is_active,
        )
        for s in sources
    ]


@router.post("", response_model=DataSource, status_code=status.HTTP_201_CREATED)
async def create_data_source(
    name: str = Form(...),
    type: DataSourceType = Form(...),
    config: str = Form(...),
    description: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    import json
    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError:
        raise ValidationError("Invalid JSON config")

    source = DataSourceModel(
        name=name,
        type=type.value,
        config=config_dict,
        description=description,
    )
    session.add(source)
    await session.flush()
    await session.refresh(source)

    logger.info("data_source_created", source_id=str(source.id), name=name)
    return DataSource(
        id=source.id,
        name=source.name,
        type=source.type,
        config=source.config,
        description=source.description,
        created_at=source.created_at,
        updated_at=source.updated_at,
        is_active=source.is_active,
    )


@router.post("/{source_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    source_id: UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    from nexalens.models.database import DataSourceModel

    source = await session.get(DataSourceModel, source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
    if source.type != DataSourceType.DOCUMENT.value:
        raise ValidationError("Data source is not a document source")

    content = await file.read()
    document = await document_service.ingest_document(session, source_id, file.filename, content)

    return {"document_id": str(document.id), "chunks": document.chunk_count}


@router.get("/{source_id}")
async def get_data_source(
    source_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    source = await session.get(DataSourceModel, source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")

    return DataSource(
        id=source.id,
        name=source.name,
        type=source.type,
        config=source.config,
        description=source.description,
        created_at=source.created_at,
        updated_at=source.updated_at,
        is_active=source.is_active,
    )


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    source_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    source = await session.get(DataSourceModel, source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")

    await session.delete(source)
    logger.info("data_source_deleted", source_id=str(source_id))