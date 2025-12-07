"""Provider synchronization logic - fetches models and updates database."""
import logging
from datetime import UTC, datetime

from shared.sources import fetch_source_models
from shared.crud import upsert_model, get_models_by_provider
from shared.models import ModelMetadata, SourceEndpoint, SourceType
from backend.litellm_client import reconcile_litellm_models

logger = logging.getLogger(__name__)


async def sync_provider(session, config, provider, push_to_litellm: bool = True) -> dict:
    """
    Sync a single provider: fetch models, update database, push to LiteLLM.

    Returns dict with stats: {"added": int, "updated": int, "orphaned": int}
    """
    stats = {"added": 0, "updated": 0, "orphaned": 0}

    try:
        # Fetch models from provider
        logger.info("Fetching models from %s at %s", provider.name, provider.base_url)

        # Create SourceEndpoint from provider
        source = SourceEndpoint(
            name=provider.name,
            base_url=provider.base_url,
            type=SourceType(provider.type),
            api_key=provider.api_key,
            prefix=provider.prefix,
            default_ollama_mode=provider.default_ollama_mode
        )

        source_models = await fetch_source_models(source)

        if not source_models or not source_models.models:
            logger.warning("No models found for provider %s", provider.name)
            return stats

        logger.info("Found %d models from %s", len(source_models.models), provider.name)

        # Track active model IDs
        active_model_ids = set()

        # Process each model
        for model_metadata in source_models.models:
            # Clean heavy Ollama payloads before storing
            if provider.type == "ollama" and hasattr(model_metadata, 'raw'):
                model_metadata.raw = _clean_ollama_payload(model_metadata.raw)

            # Upsert to database
            is_new = await upsert_model(
                session,
                provider,
                model_metadata,
                full_update=True
            )

            active_model_ids.add(model_metadata.id)

            if is_new:
                stats["added"] += 1
            else:
                stats["updated"] += 1

        # Mark orphaned models (models that no longer exist in provider)
        all_models = await get_models_by_provider(session, provider.id)
        for model in all_models:
            if model.model_id not in active_model_ids and not model.is_orphaned:
                model.is_orphaned = True
                model.orphaned_at = datetime.now(UTC)
                stats["orphaned"] += 1
                logger.info("Model %s marked as orphaned", model.model_id)

        # Reconcile with LiteLLM if configured and requested
        if push_to_litellm and config.litellm_base_url:
            await reconcile_litellm_models(
                session,
                config,
                provider,
                all_models
            )

        return stats

    except Exception as e:
        logger.error("Error syncing provider %s: %s", provider.name, e, exc_info=True)
        # Re-raise to allow transaction rollback at higher level
        raise


def _clean_ollama_payload(payload: dict) -> dict:
    """
    Remove heavy fields from Ollama model payload to save memory.

    Ollama's /api/show returns very large responses with:
    - model_info: Full tensor information
    - projector_info: Vision model projector data
    - modelfile: Complete model configuration
    - license: Full license text

    We keep only essential metadata.
    """
    if not isinstance(payload, dict):
        return payload

    cleaned = payload.copy()

    # Remove heavy fields
    heavy_fields = [
        'model_info',
        'projector_info',
        'modelfile',
        'license',
        'template',
        'system',
    ]

    for field in heavy_fields:
        cleaned.pop(field, None)

    # Keep only essential details
    if 'details' in cleaned and isinstance(cleaned['details'], dict):
        details = cleaned['details']
        # Keep family, parameter_size, quantization_level
        # Remove everything else
        essential_details = {}
        for key in ['family', 'parameter_size', 'quantization_level', 'format']:
            if key in details:
                essential_details[key] = details[key]
        cleaned['details'] = essential_details

    return cleaned
