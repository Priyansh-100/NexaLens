import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from nexalens.api.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from nexalens.models.database import UserModel
from nexalens.models.schemas import UserRole


class TestAuth:
    def test_hash_password(self):
        password = "test123"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong", hashed)

    def test_create_access_token(self):
        user = UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )
        token = create_access_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        user = UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )
        token = create_refresh_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_token(self):
        user = UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )
        token = create_access_token(user)
        payload = decode_token(token)

        assert payload.sub == str(user.id)
        assert payload.role == UserRole.ANALYST
        assert payload.type == "access"

    def test_decode_expired_token(self):
        import jwt
        from datetime import datetime, timedelta, timezone

        payload = {
            "sub": str(uuid4()),
            "role": "analyst",
            "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
            "type": "access",
        }
        expired_token = jwt.encode(payload, "test-secret", algorithm="HS256")

        with pytest.raises(Exception) as exc_info:
            decode_token(expired_token)
        assert "expired" in str(exc_info.value).lower()

    def test_decode_invalid_token(self):
        with pytest.raises(Exception) as exc_info:
            decode_token("invalid.token.here")
        assert "invalid" in str(exc_info.value).lower()


class TestTokenPayload:
    def test_token_type_access(self):
        user = UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )
        token = create_access_token(user)
        payload = decode_token(token)
        assert payload.type == "access"

    def test_token_type_refresh(self):
        user = UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )
        token = create_refresh_token(user)
        payload = decode_token(token)
        assert payload.type == "refresh"