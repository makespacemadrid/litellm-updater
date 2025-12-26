#!/bin/bash
# Safe migration script for upgrading from previous versions
set -e

echo "=== LiteLLM Companion Migration Script ==="
echo ""

# Backup database
if [ -f "/app/data/models.db" ]; then
    BACKUP_FILE="/app/data/models.db.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Creating backup: $BACKUP_FILE"
    cp /app/data/models.db "$BACKUP_FILE"
    echo "✓ Backup created"
else
    echo "ℹ No existing database found, proceeding with fresh installation"
fi

echo ""
echo "Running Alembic migrations..."
python -m alembic upgrade head || {
    echo "⚠ Alembic migrations failed, trying alternative approach..."

    echo "Initializing database schema..."
    python /app/scripts/init_database.py

    echo "Stamping database as up-to-date..."
    python -m alembic stamp head || echo "⚠ Alembic stamp failed"
}

echo ""
echo "✓ Migration complete!"
echo ""
echo "If you encounter issues, restore from backup:"
echo "  cp $BACKUP_FILE /app/data/models.db"
