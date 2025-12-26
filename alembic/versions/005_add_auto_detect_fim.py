"""Add auto_detect_fim to providers

Revision ID: 005
Revises: 004
Create Date: 2025-12-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add auto_detect_fim column with default value True
    # Check if column exists first
    import sqlite3
    from alembic import context

    config = context.config
    # Get the database URL from the config
    url = config.get_main_option("sqlalchemy.url")

    # For SQLite, extract the path
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("PRAGMA table_info(providers)")
        columns = [row[1] for row in cursor.fetchall()]

        conn.close()

        if "auto_detect_fim" not in columns:
            op.execute("""
                ALTER TABLE providers
                ADD COLUMN auto_detect_fim INTEGER NOT NULL DEFAULT 1
            """)


def downgrade() -> None:
    # SQLite doesn't support DROP COLUMN easily, would need table recreation
    pass
