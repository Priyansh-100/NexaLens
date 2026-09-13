#!/usr/bin/env python3
"""CLI commands for NexaLens administration."""

import asyncio
import sys
from uuid import uuid4

import click
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.api.auth import create_user, hash_password
from nexalens.core.config import get_settings
from nexalens.models.database import UserModel
from nexalens.models.session import async_session_maker, init_db
from nexalens.models.schemas import UserRole


@click.group()
def cli():
    """NexaLens administration commands."""
    pass


@cli.command()
@click.option("--email", prompt=True, help="Admin email")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True, help="Admin password")
@click.option("--name", prompt=True, help="Admin name")
def create_admin(email: str, password: str, name: str):
    """Create an admin user."""

    async def _create_admin():
        await init_db()
        async with async_session_maker() as session:
            try:
                user = await create_user(
                    session=session,
                    email=email,
                    password=password,
                    name=name,
                    role=UserRole.ADMIN,
                )
                await session.commit()
                click.echo(f"✓ Admin user created: {user.email} (id: {user.id})")
            except Exception as e:
                await session.rollback()
                click.echo(f"✗ Failed to create admin: {e}", err=True)
                sys.exit(1)

    asyncio.run(_create_admin())


@cli.command()
@click.option("--email", prompt=True, help="User email")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True, help="User password")
@click.option("--name", prompt=True, help="User name")
@click.option("--role", type=click.Choice([r.value for r in UserRole]), default=UserRole.VIEWER.value, help="User role")
def create_user_cmd(email: str, password: str, name: str, role: str):
    """Create a regular user."""

    async def _create_user():
        await init_db()
        async with async_session_maker() as session:
            try:
                user = await create_user(
                    session=session,
                    email=email,
                    password=password,
                    name=name,
                    role=UserRole(role),
                )
                await session.commit()
                click.echo(f"✓ User created: {user.email} (id: {user.id}, role: {user.role})")
            except Exception as e:
                await session.rollback()
                click.echo(f"✗ Failed to create user: {e}", err=True)
                sys.exit(1)

    asyncio.run(_create_user())


@cli.command()
def init_database():
    """Initialize the database (create tables)."""

    async def _init():
        await init_db()
        click.echo("✓ Database initialized")

    asyncio.run(_init())


if __name__ == "__main__":
    cli()