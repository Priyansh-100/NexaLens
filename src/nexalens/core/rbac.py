"""
Role-Based Access Control (RBAC) utilities.

Provides decorators and dependencies for enforcing role-based permissions.
"""

from functools import wraps
from typing import Callable, Optional, Set

from fastapi import Depends, HTTPException, status

from nexalens.api.auth import get_current_user
from nexalens.models.schemas import User, UserRole


class RoleRequirement:
    """Callable that enforces minimum role requirement."""

    def __init__(self, allowed_roles: Set[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[r.value for r in self.allowed_roles]}, got: {current_user.role.value}",
            )
        return current_user


# Pre-defined role requirements
require_admin = RoleRequirement({UserRole.ADMIN})
require_analyst_or_admin = RoleRequirement({UserRole.ANALYST, UserRole.ADMIN})
require_any_role = RoleRequirement({UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN})


def require_roles(*roles: UserRole) -> Callable:
    """Decorator factory for role-based access control on route functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the current_user in kwargs (injected by dependency)
            current_user = kwargs.get("current_user")
            if not current_user:
                # Try to find it in args
                for arg in args:
                    if isinstance(arg, User):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="current_user not found in request context",
                )
            
            if current_user.role not in roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {[r.value for r in roles]}, got: {current_user.role.value}",
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class Permission:
    """Represents a specific permission."""
    DATA_SOURCE_READ = "datasource:read"
    DATA_SOURCE_WRITE = "datasource:write"
    DATA_SOURCE_DELETE = "datasource:delete"
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_DELETE = "document:delete"
    QUERY_EXECUTE = "query:execute"
    FINANCIAL_MODEL = "analytics:financial_model"
    FORECAST = "analytics:forecast"
    REPORT_SCHEDULE = "reports:schedule"
    REPORT_EXECUTE = "reports:execute"
    ORGANIZATION_MANAGE = "organization:manage"
    USER_MANAGE = "user:manage"


# Role-to-permission mapping
ROLE_PERMISSIONS: dict[UserRole, Set[str]] = {
    UserRole.VIEWER: {
        Permission.DATA_SOURCE_READ,
        Permission.DOCUMENT_UPLOAD,
        Permission.QUERY_EXECUTE,
    },
    UserRole.ANALYST: {
        Permission.DATA_SOURCE_READ,
        Permission.DATA_SOURCE_WRITE,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_DELETE,
        Permission.QUERY_EXECUTE,
        Permission.FINANCIAL_MODEL,
        Permission.FORECAST,
        Permission.REPORT_SCHEDULE,
        Permission.REPORT_EXECUTE,
    },
    UserRole.ADMIN: {
        Permission.DATA_SOURCE_READ,
        Permission.DATA_SOURCE_WRITE,
        Permission.DATA_SOURCE_DELETE,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_DELETE,
        Permission.QUERY_EXECUTE,
        Permission.FINANCIAL_MODEL,
        Permission.FORECAST,
        Permission.REPORT_SCHEDULE,
        Permission.REPORT_EXECUTE,
        Permission.ORGANIZATION_MANAGE,
        Permission.USER_MANAGE,
    },
}


def has_permission(user: User, permission: str) -> bool:
    """Check if a user has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(user.role, set())


def require_permission(permission: str) -> Callable:
    """Dependency that enforces a specific permission."""
    async def _check_permission(current_user: User = Depends(get_current_user)) -> User:
        if not has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission}",
            )
        return current_user
    return _check_permission


def require_all_permissions(*permissions: str) -> Callable:
    """Dependency that enforces all specified permissions."""
    async def _check_permissions(current_user: User = Depends(get_current_user)) -> User:
        for perm in permissions:
            if not has_permission(current_user, perm):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {perm}",
                )
        return current_user
    return _check_permissions


def require_any_permission(*permissions: str) -> Callable:
    """Dependency that enforces at least one of the specified permissions."""
    async def _check_any_permission(current_user: User = Depends(get_current_user)) -> User:
        for perm in permissions:
            if has_permission(current_user, perm):
                return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: requires one of {[p for p in permissions]}",
        )
    return _check_any_permission