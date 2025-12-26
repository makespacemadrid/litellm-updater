"""Rename litellm provider type to openai and add compat type

Revision ID: 003
Revises: 002
Create Date: 2025-11-28 03:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns with direct SQL for idempotency
    try:
        op.execute("ALTER TABLE providers ADD COLUMN access_groups TEXT")
    except Exception:
        pass  # Column already exists

    try:
        op.execute("ALTER TABLE providers ADD COLUMN sync_enabled BOOLEAN NOT NULL DEFAULT 1")
    except Exception:
        pass  # Column already exists

    try:
        op.execute("ALTER TABLE models ADD COLUMN access_groups TEXT")
    except Exception:
        pass  # Column already exists

    try:
        op.execute("ALTER TABLE models ADD COLUMN sync_enabled BOOLEAN NOT NULL DEFAULT 1")
    except Exception:
        pass  # Column already exists

    # Add compat model mapping fields
    try:
        op.execute("ALTER TABLE models ADD COLUMN mapped_provider_id INTEGER")
    except Exception:
        pass  # Column already exists

    try:
        op.execute("ALTER TABLE models ADD COLUMN mapped_model_id VARCHAR")
    except Exception:
        pass  # Column already exists

    # Update existing 'litellm' provider types to 'openai'
    op.execute("UPDATE providers SET type = 'openai' WHERE type = 'litellm'")

    # Note: SQLite doesn't support DROP CONSTRAINT or ALTER COLUMN
    # Check constraints are enforced at insert time in SQLite, so we don't need to modify them
    # The server_default on sync_enabled can stay - it doesn't cause any issues


def downgrade() -> None:
    # Revert 'openai' back to 'litellm'
    op.execute("UPDATE providers SET type = 'litellm' WHERE type = 'openai'")

    # Drop the new check constraint
    op.drop_constraint('check_provider_type', 'providers', type_='check')

    # Recreate old check constraint
    op.create_check_constraint(
        'check_provider_type',
        'providers',
        "type IN ('ollama', 'litellm')"
    )

    # Drop new columns
    op.drop_column('models', 'mapped_model_id')
    op.drop_column('models', 'mapped_provider_id')
    op.drop_column('models', 'sync_enabled')
    op.drop_column('models', 'access_groups')
    op.drop_column('providers', 'sync_enabled')
    op.drop_column('providers', 'access_groups')
