from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.config import get_settings
from nexalens.core.exceptions import AuthenticationError, ValidationError
from nexalens.core.logging import get_logger
from nexalens.models.database import UserModel
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


def create_refresh_token(user: UserModel) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    payload = TokenPayload(
        sub=str(user.id),
        role=user.role,
        exp=int(expire.timestamp()),
        type="refresh",
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


def create_token_response(user: UserModel) -> Token:
    return Token(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
        expires_in=settings.access_token_expire_minutes * 60,
    )