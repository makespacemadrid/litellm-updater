"""Background sync worker that periodically syncs models from providers to LiteLLM."""
import asyncio
import logging
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.database import create_engine, init_session_maker, ensure_minimum_schema, get_database_url
from shared.crud import get_config, get_all_providers, get_provider_by_id
from backend.provider_sync import sync_provider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class _ConfigWrapper:
    """Lightweight config wrapper to avoid session-bound ORM objects."""

    def __init__(self, data: dict[str, str | None]):
        self.litellm_base_url = data.get("litellm_base_url")
        self.litellm_api_key = data.get("litellm_api_key")
        self.sync_interval_seconds = None


class SyncWorker:
    """Main sync worker that runs the sync loop."""

    def __init__(self):
        self.running = True
        self.engine = None
        self.session_maker = None

    async def initialize(self):
        """Initialize database connection."""
        logger.info("Initializing database...")

        # Create engine and session maker (using async URL)
        async_db_url = get_database_url()
        self.engine = create_engine(async_db_url)
        self.session_maker = init_session_maker(self.engine)
        # Ensure schema is up to date
        await ensure_minimum_schema(self.engine)
        logger.info("Database initialized")

    async def run(self):
        """Main sync loop."""
        logger.info("🚀 Sync worker starting...")

        # Initialize database
        await self.initialize()

        # Setup graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        logger.info("✅ Sync worker ready")

        while self.running:
            try:
                # Grab config and provider IDs in a short-lived session
                async with self.session_maker() as session:
                    config_obj = await get_config(session)
                    provider_ids = [p.id for p in await get_all_providers(session)]

                # Snapshot config values to avoid session-bound objects
                sync_interval = config_obj.sync_interval_seconds
                config_snapshot = {
                    "litellm_base_url": config_obj.litellm_base_url,
                    "litellm_api_key": config_obj.litellm_api_key,
                }

                # Check if sync is enabled
                if sync_interval == 0:
                    logger.debug("Sync disabled (interval=0), sleeping for 60s...")
                    await asyncio.sleep(60)
                    continue

                logger.info("⏰ Starting sync cycle (interval: %ds)", sync_interval)

                # Sync all enabled providers
                await self.sync_all_providers(provider_ids, config_snapshot)

                logger.info("✓ Sync cycle complete, waiting %ds for next cycle", sync_interval)

                # Wait for next interval
                await asyncio.sleep(sync_interval)

            except Exception as e:
                logger.exception("❌ Error in sync loop: %s", e)
                await asyncio.sleep(60)  # Retry after error

        logger.info("🛑 Sync worker stopped")

        # Cleanup
        if self.engine:
            await self.engine.dispose()

    async def sync_all_providers(self, provider_ids: list[int], config_snapshot: dict[str, str | None]):
        """Sync models from all enabled providers."""
        if not provider_ids:
            logger.info("No providers configured")
            return

        for provider_id in provider_ids:
            async with self.session_maker() as session:
                provider = await get_provider_by_id(session, provider_id)
                if not provider:
                    logger.debug("Provider id %s no longer exists, skipping", provider_id)
                    continue
                if provider.type == "compat":
                    logger.debug("Skipping compat provider: %s", provider.name)
                    continue
                if not provider.sync_enabled:
                    logger.debug("Skipping disabled provider: %s", provider.name)
                    continue

                try:
                    logger.info("📡 Syncing provider: %s (%s)", provider.name, provider.type)

                    # Sync this provider
                    stats = await sync_provider(session, _ConfigWrapper(config_snapshot), provider, push_to_litellm=True)

                    await session.commit()

                    logger.info(
                        "✓ Provider %s synced: %d added, %d updated, %d orphaned",
                        provider.name,
                        stats.get("added", 0),
                        stats.get("updated", 0),
                        stats.get("orphaned", 0)
                    )

                except Exception as e:
                    logger.error("❌ Failed to sync provider %s: %s", provider.name, e, exc_info=True)
                    await session.rollback()

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on SIGTERM/SIGINT."""
        logger.info("Received signal %d, shutting down gracefully...", signum)
        self.running = False


async def main():
    """Entry point."""
    worker = SyncWorker()
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
