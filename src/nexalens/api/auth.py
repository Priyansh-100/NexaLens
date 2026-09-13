from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID
import hashlib
import secrets

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.config import get_settings
from nexalens.core.exceptions import AuthenticationError, ValidationError
from nexalens.core.logging import get_logger
from nexalens.models.database import RefreshTokenModel, UserModel
from nexalens.models.session import get_db_session
from nexalens.models.schemas import Token, TokenPayload, User, UserRole

logger = get_logger(__name__)
settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user: UserModel) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = TokenPayload(
        sub=str(user.id),
        role=user.role,
        exp=int(expire.timestamp()),
        type="access",
    )
    return jwt.encode(payload.model_dump(), settings.secret_key, algorithm=ALGORITHM)


def create_refresh_token_raw() -> tuple[str, str]:
    """Generate a raw refresh token and its hash."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash


def create_refresh_token(user: UserModel, session: AsyncSession) -> str:
    """Create a new refresh token and store its hash in the database."""
    raw_token, token_hash = create_refresh_token_raw()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    
    # Store in database
    refresh_token = RefreshTokenModel(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expire,
    )
    session.add(refresh_token)
    
    payload = TokenPayload(
        sub=str(user.id),
        role=user.role,
        exp=int(expire.timestamp()),
        type="refresh",
    )
    encoded = jwt.encode(payload.model_dump(), settings.secret_key, algorithm=ALGORITHM)
    
    # Return raw token for client (they'll use this to refresh)
    return raw_token


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")


async def verify_refresh_token(session: AsyncSession, raw_token: str) -> Optional[RefreshTokenModel]:
    """Verify a refresh token and return the token record if valid."""
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    
    result = await session.execute(
        select(RefreshTokenModel).where(RefreshTokenModel.token_hash == raw_token)
    )
    token_record = result.scalar_one_or_none()
    
    if not token_record:
        return None
    
    if token_record.is_revoked:
        return None
    
    if token_record.expires_at < datetime.now(timezone.utc):
        return None
    
    return token_record


async def rotate_refresh_token(
    session: AsyncSession,
    old_token: RefreshTokenModel,
    user_id: UUID,
) -> str:
    """Rotate a refresh token: revoke old, create new."""
    # Revoke the old token
    old_token.is_revoked = True
    old_token.revoked_at = datetime.now(timezone.utc)
    
    # Get user for new token
    from nexalens.models.database import UserModel
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise AuthenticationError("User not found")
    
    # Create new refresh token
    return create_refresh_token(user, session)


async def revoke_all_user_tokens(session: AsyncSession, user_id: UUID) -> int:
    """Revoke all refresh tokens for a user. Returns count revoked."""
    result = await session.execute(
        select(RefreshTokenModel).where(
            RefreshTokenModel.user_id == user_id,
            RefreshTokenModel.is_revoked == False,
        )
    )
    tokens = result.scalars().all()
    count = 0
    for token in tokens:
        token.is_revoked = True
        token.revoked_at = datetime.now(timezone.utc)
        count += 1
    return count


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user: UserModel) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = TokenPayload(
        sub=str(user.id),
        role=user.role,
        exp=int(expire.timestamp()),
        type="access",
    )
    return jwt.encode(payload.model_dump(), settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")


async def authenticate_user(session: AsyncSession, email: str, password: str) -> Optional[UserModel]:
    result = await session.execute(select(UserModel).where(UserModel.email == email))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    session: AsyncSession = Depends(get_db_session),
    token: str = Depends(oauth2_scheme),
) -> UserModel:
    payload = decode_token(token)
    if payload.type != "access":
        raise AuthenticationError("Invalid token type")

    result = await session.execute(select(UserModel).where(UserModel.id == UUID(payload.sub)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    return user


async def create_user(
    session: AsyncSession,
    email: str,
    password: str,
    name: str,
    role: UserRole = UserRole.VIEWER,
) -> UserModel:
    existing = await session.execute(select(UserModel).where(UserModel.email == email))
    if existing.scalar_one_or_none():
        raise ValidationError("Email already registered")

    user = UserModel(
        email=email,
        name=name,
        role=role.value,
        hashed_password=hash_password(password),
    )
    session.add(user)
    await session.flush()
    return user


def create_token_response(user: UserModel) -> dict:
    """Create token response dict (not using Token schema since we need raw refresh)."""
    access_token = create_access_token(user)
    # For login, we create a raw refresh token and store it
    return {
        "access_token": access_token,
        "refresh_token": "",  # Will be set by caller after session flush
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }