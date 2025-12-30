"""Add routing group tables.

Revision ID: 006
Revises: 005
Create Date: 2026-01-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "routing_groups",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False, unique=True, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("capabilities", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "routing_targets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("group_id", sa.Integer, sa.ForeignKey("routing_groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider_id", sa.Integer, sa.ForeignKey("providers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_id", sa.String, nullable=False),
        sa.Column("weight", sa.Integer, nullable=False, server_default="1"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("group_id", "provider_id", "model_id", name="uq_routing_target"),
    )
    op.create_index("ix_routing_targets_group_id", "routing_targets", ["group_id"])
    op.create_index("ix_routing_targets_provider_id", "routing_targets", ["provider_id"])
    op.create_index("ix_routing_targets_priority", "routing_targets", ["priority"])

    op.create_table(
        "routing_provider_limits",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("group_id", sa.Integer, sa.ForeignKey("routing_groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider_id", sa.Integer, sa.ForeignKey("providers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("max_requests_per_hour", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("group_id", "provider_id", name="uq_routing_provider_limit"),
    )
    op.create_index("ix_routing_provider_limits_group_id", "routing_provider_limits", ["group_id"])
    op.create_index("ix_routing_provider_limits_provider_id", "routing_provider_limits", ["provider_id"])


def downgrade() -> None:
    op.drop_index("ix_routing_provider_limits_provider_id", table_name="routing_provider_limits")
    op.drop_index("ix_routing_provider_limits_group_id", table_name="routing_provider_limits")
    op.drop_table("routing_provider_limits")

    op.drop_index("ix_routing_targets_priority", table_name="routing_targets")
    op.drop_index("ix_routing_targets_provider_id", table_name="routing_targets")
    op.drop_index("ix_routing_targets_group_id", table_name="routing_targets")
    op.drop_table("routing_targets")

    op.drop_table("routing_groups")
