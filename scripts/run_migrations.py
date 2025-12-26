#!/usr/bin/env python3
"""Run Alembic migrations automatically."""
import logging
from pathlib import Path
from alembic.config import Config
from alembic import command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migrations():
    """Run all pending Alembic migrations."""
    try:
        # Get the alembic.ini path
        base_dir = Path(__file__).parent.parent
        alembic_ini = base_dir / "alembic.ini"

        if not alembic_ini.exists():
            logger.warning(f"Alembic config not found at {alembic_ini}, skipping migrations")
            return

        logger.info("Running Alembic migrations...")
        alembic_cfg = Config(str(alembic_ini))
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrations completed successfully")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        # Don't raise - fallback to ensure_minimum_schema

if __name__ == "__main__":
    run_migrations()
