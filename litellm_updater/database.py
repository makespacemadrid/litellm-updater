"""Database setup and connection management."""
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

DEFAULT_DB_PATH = Path("data/models.db")

# Global session maker (initialized in lifespan)
async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_database_url(path: Path = DEFAULT_DB_PATH) -> str:
    """Return SQLite async connection string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path}"


def create_engine(db_url: str | None = None) -> AsyncEngine:
    """Create async SQLAlchemy engine."""
    if db_url is None:
        db_url = get_database_url()
    return create_async_engine(db_url, echo=False, future=True)


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


def init_session_maker(engine: AsyncEngine) -> None:
    """Initialize the global session maker."""
    global async_session_maker
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
