"""Database setup and connection management."""
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError

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
    """Create async SQLAlchemy engine."""
    if db_url is None:
        db_url = get_database_url()
    return create_async_engine(db_url, echo=False, future=True)


def run_migrations(
    db_url: str | None = None,
    config_path: Path | str | None = None,
    script_location: Path | str | None = None,
) -> None:
    """Apply alembic migrations up to head."""

    base_dir = Path(__file__).resolve().parent.parent
    db_url = db_url or get_database_url()
    config_path = Path(config_path) if config_path else base_dir / "alembic.ini"
    script_location = Path(script_location) if script_location else base_dir / "alembic"

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

        # Models.system_tags / user_tags / access_groups / sync_enabled
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

        # Normalize default values once columns exist
        await conn.exec_driver_sql(
            "UPDATE models SET system_tags='[]' WHERE system_tags IS NULL"
        )


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
