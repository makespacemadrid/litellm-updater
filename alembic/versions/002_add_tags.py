"""Add provider and model tags

Revision ID: 002
Revises: 001
Create Date: 2025-11-28 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns with direct SQL to avoid transaction issues
    # Use try/except for idempotency (columns might already exist)

    try:
        op.execute("ALTER TABLE providers ADD COLUMN tags TEXT")
    except Exception:
        pass  # Column already exists

    try:
        op.execute("ALTER TABLE models ADD COLUMN system_tags TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass  # Column already exists

    try:
        op.execute("ALTER TABLE models ADD COLUMN user_tags TEXT")
    except Exception:
        pass  # Column already exists

    # Backfill system_tags for existing rows (safe to run multiple times)
    op.execute("UPDATE models SET system_tags = '[]' WHERE system_tags IS NULL OR system_tags = ''")


def downgrade() -> None:
    op.drop_column('models', 'user_tags')
    op.drop_column('models', 'system_tags')
    op.drop_column('providers', 'tags')
