#!/usr/bin/env python3
"""
Initialize read-only PostgreSQL role for safe SQL execution.

This script creates:
1. A `nexalens_readonly` role with SELECT-only permissions
2. Grants SELECT on all current and future tables in public schema
3. Optionally creates a dedicated user for external connections

Run after database initialization:
    python scripts/init_readonly_role.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from nexalens.core.config import get_settings


async def init_readonly_role():
    settings = get_settings()
    
    # Use the main database URL
    engine = create_async_engine(settings.database_url, echo=True)
    
    async with engine.begin() as conn:
        # Create read-only role
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'nexalens_readonly') THEN
                    CREATE ROLE nexalens_readonly NOLOGIN;
                END IF;
            END
            $$;
        """))
        print("✓ Role 'nexalens_readonly' created/verified")

        # Grant CONNECT on database
        await conn.execute(text("GRANT CONNECT ON DATABASE nexalens TO nexalens_readonly;"))
        print("✓ GRANT CONNECT ON DATABASE")

        # Grant USAGE on schema
        await conn.execute(text("GRANT USAGE ON SCHEMA public TO nexalens_readonly;"))
        print("✓ GRANT USAGE ON SCHEMA public")

        # Grant SELECT on all existing tables
        await conn.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO nexalens_readonly;"))
        print("✓ GRANT SELECT ON ALL TABLES IN SCHEMA public")

        # Grant SELECT on all sequences (for auto-increment columns)
        await conn.execute(text("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nexalens_readonly;"))
        print("✓ GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public")

        # Set default privileges for future tables
        await conn.execute(text("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT ON TABLES TO nexalens_readonly;
        """))
        print("✓ ALTER DEFAULT PRIVILEGES for future tables")

        await conn.execute(text("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT ON SEQUENCES TO nexalens_readonly;
        """))
        print("✓ ALTER DEFAULT PRIVILEGES for future sequences")

        # Create a dedicated user for read-only connections (optional)
        readonly_password = os.getenv("READONLY_USER_PASSWORD", "readonly-change-me")
        await conn.execute(text(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'nexalens_ro_user') THEN
                    CREATE ROLE nexalens_ro_user LOGIN PASSWORD '{readonly_password}' IN ROLE nexalens_readonly;
                ELSE
                    ALTER ROLE nexalens_ro_user WITH PASSWORD '{readonly_password}';
                END IF;
            END
            $$;
        """))
        print("✓ User 'nexalens_ro_user' created/updated")

        # Set statement timeout for the role (30 seconds)
        await conn.execute(text("ALTER ROLE nexalens_ro_user SET statement_timeout = '30s';"))
        print("✓ statement_timeout set to 30s for nexalens_ro_user")

        # Also set for the role itself
        await conn.execute(text("ALTER ROLE nexalens_readonly SET statement_timeout = '30s';"))
        print("✓ statement_timeout set to 30s for nexalens_readonly")

    await engine.dispose()
    print("\n✅ Read-only role initialization complete!")
    print(f"\nUse these credentials for read-only connections:")
    print(f"  User: nexalens_ro_user")
    print(f"  Password: {readonly_password}")
    print(f"\n⚠️  Change the password in production via READONLY_USER_PASSWORD env var")


if __name__ == "__main__":
    import os
    asyncio.run(init_readonly_role())