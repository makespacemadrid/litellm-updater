"""Initial schema with providers and models tables

Revision ID: 001
Revises:
Create Date: 2025-11-28 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create providers table
    op.create_table(
        'providers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('base_url', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('api_key', sa.String(), nullable=True),
        sa.Column('prefix', sa.String(), nullable=True),
        sa.Column('default_ollama_mode', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint("type IN ('ollama', 'litellm')", name='check_provider_type'),
        sa.CheckConstraint(
            "default_ollama_mode IS NULL OR default_ollama_mode IN ('ollama', 'openai')",
            name='check_default_ollama_mode'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_providers_name'), 'providers', ['name'], unique=False)

    # Create models table
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('provider_id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=True),
        sa.Column('context_window', sa.Integer(), nullable=True),
        sa.Column('max_input_tokens', sa.Integer(), nullable=True),
        sa.Column('max_output_tokens', sa.Integer(), nullable=True),
        sa.Column('max_tokens', sa.Integer(), nullable=True),
        sa.Column('capabilities', sa.Text(), nullable=True),
        sa.Column('litellm_params', sa.Text(), nullable=False),
        sa.Column('raw_metadata', sa.Text(), nullable=False),
        sa.Column('user_params', sa.Text(), nullable=True),
        sa.Column('ollama_mode', sa.String(), nullable=True),
        sa.Column('first_seen', sa.DateTime(), nullable=False),
        sa.Column('last_seen', sa.DateTime(), nullable=False),
        sa.Column('is_orphaned', sa.Boolean(), nullable=False),
        sa.Column('orphaned_at', sa.DateTime(), nullable=True),
        sa.Column('user_modified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "ollama_mode IS NULL OR ollama_mode IN ('ollama', 'openai')",
            name='check_model_ollama_mode'
        ),
        sa.ForeignKeyConstraint(['provider_id'], ['providers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider_id', 'model_id')
    )
    op.create_index(op.f('ix_models_provider_id'), 'models', ['provider_id'], unique=False)
    op.create_index(op.f('ix_models_model_id'), 'models', ['model_id'], unique=False)
    op.create_index(op.f('ix_models_is_orphaned'), 'models', ['is_orphaned'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_models_is_orphaned'), table_name='models')
    op.drop_index(op.f('ix_models_model_id'), table_name='models')
    op.drop_index(op.f('ix_models_provider_id'), table_name='models')
    op.drop_table('models')
    op.drop_index(op.f('ix_providers_name'), table_name='providers')
    op.drop_table('providers')
