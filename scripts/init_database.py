#!/usr/bin/env python3
"""Initialize database schema before service starts."""
import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db_models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_database():
    """Create database and tables, then ensure auto_detect_fim column exists."""
    db_path = Path("/app/data/models.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Database path: {db_path.absolute()}")
    logger.info(f"Database exists: {db_path.exists()}")
    if db_path.exists():
        logger.info(f"Database size: {db_path.stat().st_size} bytes")

    # Create async engine
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        connect_args={"timeout": 30}
    )

    logger.info("Creating database tables...")
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        # Explicit commit by exiting the context manager

    logger.info("Database tables created successfully")

    # Important: Dispose the async engine and wait a moment
    # to ensure all async writes are flushed to disk
    await engine.dispose()
    await asyncio.sleep(0.5)

    # Now use direct SQLite to verify and add column if needed
    # This is a failsafe in case the column wasn't in the model when tables were created
    logger.info("Verifying auto_detect_fim column...")
    logger.info(f"Opening SQLite connection to: {db_path.absolute()}")
    try:
        # Open connection with immediate mode to avoid WAL conflicts
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        cursor = conn.cursor()
        logger.info("SQLite connection opened successfully")

        # Check if column exists
        cursor.execute("PRAGMA table_info(providers)")
        columns = [row[1] for row in cursor.fetchall()]

        logger.info(f"Current providers table columns: {columns}")

        if "auto_detect_fim" not in columns:
            logger.info("Adding auto_detect_fim column...")
            cursor.execute(
                "ALTER TABLE providers ADD COLUMN auto_detect_fim INTEGER NOT NULL DEFAULT 1"
            )
            # Force synchronous write to disk
            cursor.execute("PRAGMA synchronous = FULL")
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("auto_detect_fim column added successfully")

            # Verify it was added
            cursor.execute("PRAGMA table_info(providers)")
            columns_after = [row[1] for row in cursor.fetchall()]
            logger.info(f"Providers table columns after add: {columns_after}")

            if "auto_detect_fim" not in columns_after:
                raise RuntimeError("auto_detect_fim column was not added despite successful ALTER TABLE!")
        else:
            logger.info("auto_detect_fim column already exists")

        # Force all changes to disk
        cursor.execute("PRAGMA synchronous = FULL")
        logger.info("Running VACUUM to reorganize database...")
        cursor.execute("VACUUM")  # Force complete database reorganization
        conn.commit()  # Explicit commit even though isolation_level=None
        conn.close()
        logger.info(f"Database closed. Final size: {db_path.stat().st_size} bytes")

        # Give filesystem time to sync
        time.sleep(0.5)

        # Final verification with new connection
        logger.info("Final verification of auto_detect_fim column...")
        verify_conn = sqlite3.connect(str(db_path))
        verify_cursor = verify_conn.cursor()
        verify_cursor.execute("PRAGMA table_info(providers)")
        final_columns = [row[1] for row in verify_cursor.fetchall()]
        verify_conn.close()

        if "auto_detect_fim" not in final_columns:
            raise RuntimeError(f"FATAL: auto_detect_fim column not found after all attempts! Columns: {final_columns}")
        else:
            logger.info(f"✓ Verified: auto_detect_fim column exists. All columns: {final_columns}")
    except Exception as e:
        logger.error(f"Error ensuring auto_detect_fim column: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_database())
