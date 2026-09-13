from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.api.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_refresh_token,
    rotate_refresh_token,
    create_user,
    get_current_user,
)
from nexalens.models.schemas import Token, User, UserRole
from nexalens.models.session import get_db_session

router = APIRouter(prefix="/auth", tags=["authentication"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: UserRole = UserRole.VIEWER


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_db_session),
):
    user = await create_user(session, request.email, request.password, request.name, request.role)
    await session.flush()
    
    # Create tokens
    from nexalens.api.auth import create_access_token, create_refresh_token
    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user, session)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=60 * 60,  # 60 minutes
    )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db_session),
):
    from nexalens.api.auth import authenticate_user, create_access_token, create_refresh_token
    
    user = await authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    await session.flush()
    
    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user, session)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=60 * 60,
    )


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


class RefreshTokenRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_db_session),
):
    from nexalens.api.auth import verify_refresh_token, rotate_refresh_token, create_access_token
    from nexalens.models.database import UserModel
    from nexalens.core.exceptions import AuthenticationError
    
    # Verify the refresh token
    token_record = await verify_refresh_token(session, request.refresh_token)
    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Get user
    from nexalens.models.database import UserModel
    from sqlalchemy import select
    result = await session.execute(select(UserModel).where(UserModel.id == token_record.user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Rotate the refresh token
    new_refresh_token = await rotate_refresh_token(session, token_record, user.id)
    await session.commit()
    
    # Create new access token
    from nexalens.api.auth import create_access_token
    access_token = create_access_token(user)
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=60 * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """Revoke all refresh tokens for the current user."""
    from nexalens.api.auth import revoke_all_user_tokens
    await revoke_all_user_tokens(session, current_user.id)
    await session.commit()