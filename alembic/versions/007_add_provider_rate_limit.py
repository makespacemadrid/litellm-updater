"""Add provider max_requests_per_hour.

Revision ID: 007
Revises: 006
Create Date: 2026-01-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("providers", sa.Column("max_requests_per_hour", sa.Integer, nullable=True))


def downgrade() -> None:
    # SQLite doesn't support DROP COLUMN; leave as-is.
    pass
