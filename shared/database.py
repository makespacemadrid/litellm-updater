"""Database setup and connection management."""
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/models.db")

# Global session maker (initialized in lifespan)
async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_database_url(path: Path = DEFAULT_DB_PATH) -> str:
    """Return SQLite async connection string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path}"


def get_sync_database_url(path: Path = DEFAULT_DB_PATH) -> str:
    """Return SQLite sync connection string (used by Alembic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def create_engine(db_url: str | None = None) -> AsyncEngine:
    """Create async SQLAlchemy engine with WAL mode for SQLite."""
    if db_url is None:
        db_url = get_database_url()

    # Configure connection args for SQLite with WAL mode and timeouts
    connect_args = {}
    if "sqlite" in db_url:
        connect_args = {
            "check_same_thread": False,
            "timeout": 30.0,  # Wait up to 30s for locks
        }

    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
        connect_args=connect_args,
    )

    return engine


def run_migrations(
    db_url: str | None = None,
    config_path: Path | str | None = None,
    script_location: Path | str | None = None,
) -> None:
    """Apply alembic migrations up to head."""
    import sqlite3

    base_dir = Path(__file__).resolve().parent.parent
    db_url = db_url or get_sync_database_url()
    config_path = Path(config_path) if config_path else base_dir / "alembic.ini"
    script_location = Path(script_location) if script_location else base_dir / "alembic"

    # Check if alembic_version table exists and has the latest version
    # to avoid blocking on already-migrated databases
    if db_url.startswith("sqlite"):
        db_path = db_url.replace("sqlite:///", "")
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT version_num FROM alembic_version LIMIT 1")
            version = cursor.fetchone()
            conn.close()
            if version:
                logger.info(f"Database already at version {version[0]}, skipping migrations")
                return
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            # Table doesn't exist or can't be queried, continue with migrations
            pass

    alembic_config = Config(str(config_path))
    alembic_config.set_main_option("script_location", str(script_location))
    alembic_config.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_config, "head")


async def ensure_minimum_schema(engine: AsyncEngine) -> None:
    """
    Ensure required columns exist when migrations are unavailable.

    This is a safety net for packaged/embedded deployments where the alembic
    scripts may not be present. It creates tables if they don't exist,
    then uses lightweight ALTER TABLE statements to add missing columns
    introduced after the initial schema.
    """
    from .db_models import Base

    async with engine.begin() as conn:
        # First, create all tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)

        # Providers.tags / access_groups / sync_enabled
        result = await conn.exec_driver_sql("PRAGMA table_info(providers)")
        provider_columns = {row[1] for row in result}
        if "tags" not in provider_columns:
            await conn.exec_driver_sql("ALTER TABLE providers ADD COLUMN tags TEXT")
        if "access_groups" not in provider_columns:
            await conn.exec_driver_sql("ALTER TABLE providers ADD COLUMN access_groups TEXT")
        if "sync_enabled" not in provider_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE providers ADD COLUMN sync_enabled INTEGER NOT NULL DEFAULT 1"
            )

        # Models.system_tags / user_tags / access_groups / sync_enabled / mapped_provider_id / mapped_model_id
        result = await conn.exec_driver_sql("PRAGMA table_info(models)")
        model_columns = {row[1] for row in result}
        if "system_tags" not in model_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE models ADD COLUMN system_tags TEXT NOT NULL DEFAULT '[]'"
            )
        if "user_tags" not in model_columns:
            await conn.exec_driver_sql("ALTER TABLE models ADD COLUMN user_tags TEXT")
        if "access_groups" not in model_columns:
            await conn.exec_driver_sql("ALTER TABLE models ADD COLUMN access_groups TEXT")
        if "sync_enabled" not in model_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE models ADD COLUMN sync_enabled INTEGER NOT NULL DEFAULT 1"
            )
        if "mapped_provider_id" not in model_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE models ADD COLUMN mapped_provider_id INTEGER"
            )
        if "mapped_model_id" not in model_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE models ADD COLUMN mapped_model_id VARCHAR"
            )

        # Normalize default values once columns exist
        await conn.exec_driver_sql(
            "UPDATE models SET system_tags='[]' WHERE system_tags IS NULL"
        )

        # Update provider type constraint to allow 'compat' type
        # SQLite doesn't support ALTER CHECK CONSTRAINT, so we need to check if the constraint allows 'compat'
        # If not, we need to recreate the table (this is a one-time operation)
        try:
            result = await conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='providers'"
            )
            table_sql = result.fetchone()[0]
            columns_result = await conn.exec_driver_sql("PRAGMA table_info(providers)")
            column_names = [row[1] for row in columns_result]
            expected_columns = [
                "id",
                "name",
                "base_url",
                "type",
                "api_key",
                "prefix",
                "default_ollama_mode",
                "tags",
                "access_groups",
                "sync_enabled",
                "created_at",
                "updated_at",
            ]
            missing_compat = "'compat'" not in table_sql and '"compat"' not in table_sql
            missing_ollama_chat = "ollama_chat" not in table_sql
            wrong_order = column_names != expected_columns
            if missing_compat or missing_ollama_chat or wrong_order:
                # Need to recreate providers table with updated constraint
                await conn.exec_driver_sql("PRAGMA foreign_keys=off")
                await conn.exec_driver_sql(
                    """
                    CREATE TABLE providers_new (
                        id INTEGER NOT NULL,
                        name VARCHAR NOT NULL,
                        base_url VARCHAR NOT NULL,
                        type VARCHAR NOT NULL,
                        api_key VARCHAR,
                        prefix VARCHAR,
                        default_ollama_mode VARCHAR,
                        tags TEXT,
                        access_groups TEXT,
                        sync_enabled BOOLEAN NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        PRIMARY KEY (id),
                        CONSTRAINT check_provider_type CHECK (type IN ('ollama', 'openai', 'compat')),
                        CONSTRAINT check_default_ollama_mode CHECK (default_ollama_mode IS NULL OR default_ollama_mode IN ('ollama', 'ollama_chat', 'openai'))
                    )
                    """
                )
                # Copy data explicitly by column to avoid misalignment
                await conn.exec_driver_sql(
                    """
                    INSERT INTO providers_new (
                        id, name, base_url, type, api_key, prefix, default_ollama_mode,
                        tags, access_groups, sync_enabled, created_at, updated_at
                    )
                    SELECT
                        id, name, base_url, type, api_key, prefix, default_ollama_mode,
                        tags, access_groups, sync_enabled, created_at, updated_at
                    FROM providers
                    """
                )
                # Drop old table
                await conn.exec_driver_sql("DROP TABLE providers")
                # Rename new table
                await conn.exec_driver_sql("ALTER TABLE providers_new RENAME TO providers")
                await conn.exec_driver_sql("PRAGMA foreign_keys=on")

                # Normalize datetime columns that may have been stored as integers during migration
                await conn.exec_driver_sql(
                    "UPDATE providers SET created_at=updated_at WHERE typeof(created_at)='integer'"
                )
        except Exception:
            # If constraint update fails, log but don't crash
            logger.exception("Failed to update provider type constraint")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.

    Usage in FastAPI:
        @app.get("/endpoint")
        async def endpoint(session: AsyncSession = Depends(get_session)):
            ...
    """
    if async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() in lifespan.")

    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Initialize the global session maker and return it."""
    global async_session_maker
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return async_session_maker
