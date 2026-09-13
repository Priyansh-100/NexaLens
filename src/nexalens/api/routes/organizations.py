from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from nexalens.api.auth import get_current_user
from nexalens.core.exceptions import ValidationError
from nexalens.core.logging import get_logger
from nexalens.core.rbac import require_admin
from nexalens.models.database import OrganizationModel
from nexalens.models.schemas import (
    Organization,
    OrganizationCreate,
    OrganizationUpdate,
    User,
)
from nexalens.models.session import get_db_session

router = APIRouter(prefix="/organizations", tags=["organizations"])
logger = get_logger(__name__)


@router.get("", response_model=list[Organization])
async def list_organizations(
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> list[Organization]:
    result = await session.execute(select(OrganizationModel).where(OrganizationModel.is_active == True))
    orgs = result.scalars().all()
    return [
        Organization(
            id=o.id,
            name=o.name,
            slug=o.slug,
            description=o.description,
            is_active=o.is_active,
            created_at=o.created_at,
            updated_at=o.updated_at,
        )
        for o in orgs
    ]


@router.post("", response_model=Organization, status_code=status.HTTP_201_CREATED)
async def create_organization(
    org_in: OrganizationCreate,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_admin),
) -> Organization:
    # Check if slug already exists
    existing = await session.execute(select(OrganizationModel).where(OrganizationModel.slug == org_in.slug))
    if existing.scalar_one_or_none():
        raise ValidationError(f"Organization with slug '{org_in.slug}' already exists")

    org = OrganizationModel(
        name=org_in.name,
        slug=org_in.slug,
        description=org_in.description,
    )
    session.add(org)
    await session.flush()
    await session.refresh(org)

    logger.info("organization_created", org_id=str(org.id), name=org.name)
    return Organization(
        id=org.id,
        name=org.name,
        slug=org.slug,
        description=org.description,
        is_active=org.is_active,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


@router.get("/{org_id}", response_model=Organization)
async def get_organization(
    org_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> Organization:
    org = await session.get(OrganizationModel, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    return Organization(
        id=org.id,
        name=org.name,
        slug=org.slug,
        description=org.description,
        is_active=org.is_active,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


@router.patch("/{org_id}", response_model=Organization)
async def update_organization(
    org_id: UUID,
    updates: OrganizationUpdate,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_admin),
) -> Organization:
    org = await session.get(OrganizationModel, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(org, field, value)

    await session.flush()
    await session.refresh(org)

    logger.info("organization_updated", org_id=str(org_id))
    return Organization(
        id=org.id,
        name=org.name,
        slug=org.slug,
        description=org.description,
        is_active=org.is_active,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


@router.delete("/{org_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization(
    org_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_admin),
):
    org = await session.get(OrganizationModel, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    await session.delete(org)
    logger.info("organization_deleted", org_id=str(org_id))