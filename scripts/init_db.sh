#!/bin/bash
set -e

echo "🔧 Initializing database..."

PGPASSWORD=postgres psql -h localhost -U postgres -d nexalens -c "
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"

echo "✅ Database initialized"