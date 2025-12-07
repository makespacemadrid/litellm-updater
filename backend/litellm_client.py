"""LiteLLM API client for pushing models."""
import logging
import httpx

from shared.models import ModelMetadata
from shared.sources import _make_auth_headers, DEFAULT_TIMEOUT
from shared.tags import generate_model_tags

logger = logging.getLogger(__name__)


async def reconcile_litellm_models(session, config, provider, models, remove_orphaned=True):
    """
    Reconcile models with LiteLLM: add missing, update changed, optionally remove orphaned.

    Args:
        session: Database session
        config: App configuration
        provider: Provider model
        models: List of models from database for this provider
        remove_orphaned: If True, delete models from LiteLLM that were created by updater
                        but are no longer in the database. Should be True for scheduled syncs,
                        False for manual pushes.
    """
    logger.info(f"Reconciling provider={provider.name}, models_count={len(models)}, remove_orphaned={remove_orphaned}")

    stats = {"added": 0, "updated": 0, "deleted": 0, "duplicates_removed": 0, "skipped": 0}

    if not config.litellm_base_url:
        logger.debug("LiteLLM not configured, skipping reconciliation")
        return

    try:
        async with httpx.AsyncClient() as client:
            # Get current models from LiteLLM
            litellm_models = await fetch_litellm_models(
                client,
                config.litellm_base_url,
                config.litellm_api_key
            )
            logger.info(f"Found {len(litellm_models)} models in LiteLLM")

            # Index by unique_id tag
            litellm_index = {}
            models_without_unique_id = 0
            duplicates_to_prune = []
            for m in litellm_models:
                # Prefer tags from litellm_params but also look at model_info/root to avoid missing unique_id
                tags = m.get('litellm_params', {}).get('tags', [])
                model_info_tags = m.get('model_info', {}).get('tags', [])
                root_tags = m.get('tags', [])
                combined_tags = list(tags or []) + list(model_info_tags or []) + list(root_tags or [])
                unique_id_tag = next((t for t in combined_tags if isinstance(t, str) and t.startswith('unique_id:')), None)
                if unique_id_tag:
                    unique_id_tag = unique_id_tag.lower()
                    if unique_id_tag in litellm_index:
                        duplicates_to_prune.append(m)
                    else:
                        litellm_index[unique_id_tag] = m
                else:
                    models_without_unique_id += 1

            logger.info(f"Indexed {len(litellm_index)} models by unique_id from LiteLLM ({models_without_unique_id} models without unique_id tag)")
            if litellm_index:
                sample_keys = list(litellm_index.keys())[:3]
                logger.info(f"Sample unique_ids from LiteLLM: {sample_keys}")

            provider_unique_prefix = f"unique_id:{provider.name.lower()}/"

            # Remove duplicate entries that share the same unique_id
            for dup in duplicates_to_prune:
                model_id = dup.get('model_info', {}).get('id')
                if model_id:
                    try:
                        await delete_model_from_litellm(
                            client,
                            config.litellm_base_url,
                            config.litellm_api_key,
                            model_id
                        )
                        stats["duplicates_removed"] += 1
                        logger.warning("Pruned duplicate LiteLLM entry id=%s", model_id)
                    except Exception as exc:
                        logger.warning("Failed pruning duplicate id=%s: %s", model_id, exc)

            # Build set of unique_ids from our database models
            db_unique_ids = set()
            for model in models:
                if not model.is_orphaned and model.sync_enabled:
                    unique_id_tag = f"unique_id:{provider.name}/{model.model_id}".lower()
                    db_unique_ids.add(unique_id_tag)

            logger.info(f"Built {len(db_unique_ids)} unique_ids from database")
            if db_unique_ids:
                sample_db_ids = list(db_unique_ids)[:3]
                logger.info(f"Sample unique_ids from DB: {sample_db_ids}")

            # Add or update models
            for model in models:
                if model.is_orphaned or not model.sync_enabled:
                    stats["skipped"] += 1
                    continue

                unique_id_tag = f"unique_id:{provider.name}/{model.model_id}".lower()

                if unique_id_tag not in litellm_index:
                    # Model doesn't exist in LiteLLM, add it
                    await push_model_to_litellm(
                        client,
                        config.litellm_base_url,
                        config.litellm_api_key,
                        provider,
                        model
                    )
                    stats["added"] += 1
                    logger.info("Added model %s to LiteLLM", model.model_id)
                else:
                    # Model exists, check if update needed
                    litellm_model = litellm_index[unique_id_tag]
                    if _needs_update(provider, model, litellm_model):
                        await update_model_in_litellm(
                            client,
                            config.litellm_base_url,
                            config.litellm_api_key,
                            provider,
                            model,
                            litellm_model
                        )
                        stats["updated"] += 1
                        logger.info("Updated model %s in LiteLLM", model.model_id)
                    else:
                        stats["skipped"] += 1

            # Remove orphaned models created by updater (only during scheduled sync)
            if remove_orphaned:
                for unique_id_tag, litellm_model in litellm_index.items():
                    # Only prune models that belong to this provider
                    if not unique_id_tag.startswith(provider_unique_prefix):
                        continue
                    created_by = litellm_model.get('model_info', {}).get('created_by')
                    if created_by == 'updater' and unique_id_tag not in db_unique_ids:
                        # Model was created by updater but no longer in database
                        model_id = litellm_model.get('model_info', {}).get('id')
                        if model_id:
                            await delete_model_from_litellm(
                                client,
                                config.litellm_base_url,
                                config.litellm_api_key,
                                model_id
                            )
                            stats["deleted"] += 1
                            logger.info("Deleted orphaned model %s from LiteLLM", unique_id_tag)

    except Exception as e:
        logger.error("Failed to reconcile with LiteLLM: %s", e)
        # Don't raise - sync should continue even if LiteLLM push fails
    return stats

async def fetch_litellm_models(client: httpx.AsyncClient, base_url: str, api_key: str | None):
    """Fetch all models from LiteLLM."""
    url = f"{base_url.rstrip('/')}/model/info"
    headers = _make_auth_headers(api_key)

    try:
        response = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except httpx.HTTPStatusError as e:
        # LiteLLM returns 500 when there are no models
        if e.response.status_code == 500:
            logger.warning("LiteLLM returned 500 (likely no models exist yet)")
            return []
        raise


async def push_model_to_litellm(client: httpx.AsyncClient, base_url: str, api_key: str | None, provider, model):
    """Push a single model to LiteLLM."""
    # Build litellm_params
    litellm_params = _build_litellm_params(provider, model)

    # Build model_info
    model_info = model.effective_params.copy()

    # Set litellm_provider
    ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama"

    if provider.type == "openai":
        model_info["litellm_provider"] = "openai"
    elif provider.type == "ollama":
        model_info["mode"] = ollama_mode
        model_info["litellm_provider"] = "openai" if ollama_mode == "openai" else "ollama"

    # Generate tags
    from shared.models import ModelMetadata as PydanticModelMetadata

    metadata = PydanticModelMetadata.from_raw(model.model_id, model.raw_metadata_dict)
    tags = generate_model_tags(
        provider_name=provider.name,
        provider_type=provider.type,
        metadata=metadata,
        provider_tags=provider.tags_list,
        mode=ollama_mode if provider.type != "compat" else None
    )

    litellm_params["tags"] = tags
    model_info["tags"] = tags

    # Add access_groups
    access_groups = model.get_effective_access_groups()
    if access_groups:
        model_info["access_groups"] = access_groups

    # Mark as created/updated by updater with timestamp
    from datetime import datetime, UTC
    current_time = datetime.now(UTC)
    model_info["created_by"] = "updater"
    model_info["updated_at"] = current_time.isoformat()

    # Build display name
    display_name = model.get_display_name(apply_prefix=True)

    # Push to LiteLLM
    url = f"{base_url.rstrip('/')}/model/new"
    headers = _make_auth_headers(api_key)

    payload = {
        "model_name": display_name,
        "litellm_params": litellm_params,
        "model_info": model_info
    }

    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


async def update_model_in_litellm(client: httpx.AsyncClient, base_url: str, api_key: str | None, provider, model, litellm_model):
    """Update an existing model in LiteLLM."""
    # First delete the old model
    model_id = litellm_model.get('model_info', {}).get('id')
    if not model_id:
        logger.warning("Cannot update model without id, skipping")
        return

    await delete_model_from_litellm(client, base_url, api_key, model_id)

    # Then create the new version
    await push_model_to_litellm(client, base_url, api_key, provider, model)


async def delete_model_from_litellm(client: httpx.AsyncClient, base_url: str, api_key: str | None, model_id: str):
    """Delete a model from LiteLLM."""
    url = f"{base_url.rstrip('/')}/model/delete"
    headers = _make_auth_headers(api_key)

    payload = {"id": model_id}

    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


def _needs_update(provider, model, litellm_model) -> bool:
    """
    Check if a model in LiteLLM needs to be updated based on database model.

    Compares key fields to determine if re-sync is needed.
    """
    # Get current values from database model
    db_display_name = model.get_display_name(apply_prefix=True)
    db_params = _build_litellm_params(provider, model)
    db_model_str = db_params.get('model', '')
    db_api_base = db_params.get('api_base', '')

    # Get current values from LiteLLM
    ll_model_name = litellm_model.get('model_name', '')
    ll_params = litellm_model.get('litellm_params', {})
    ll_model_str = ll_params.get('model', '')
    ll_api_base = ll_params.get('api_base', '')

    # Compare key fields
    if db_display_name != ll_model_name:
        logger.debug("Model name changed: %s != %s", db_display_name, ll_model_name)
        return True

    if db_model_str != ll_model_str:
        logger.debug("Model string changed: %s != %s", db_model_str, ll_model_str)
        return True

    if db_api_base != ll_api_base:
        logger.debug("API base changed: %s != %s", db_api_base, ll_api_base)
        return True

    # Check if access groups changed
    db_access_groups = set(model.get_effective_access_groups())
    ll_access_groups = set(litellm_model.get('model_info', {}).get('access_groups', []))
    if db_access_groups != ll_access_groups:
        logger.debug("Access groups changed: %s != %s", db_access_groups, ll_access_groups)
        return True

    return False


def _build_litellm_params(provider, model) -> dict:
    """Build litellm_params for a model."""
    litellm_params = {}
    model_id = model.model_id

    ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama_chat"

    if provider.type == "openai":
        litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_base"] = provider.base_url
        litellm_params["api_key"] = provider.api_key or "sk-1234"
    elif provider.type == "ollama":
        if ollama_mode == "openai":
            litellm_params["model"] = f"openai/{model_id}"
            api_base = provider.base_url.rstrip("/")
            litellm_params["api_base"] = f"{api_base}/v1"
            litellm_params["api_key"] = provider.api_key or "sk-1234"
        elif ollama_mode == "ollama":
            litellm_params["model"] = f"ollama/{model_id}"
            litellm_params["api_base"] = provider.base_url
        else:
            # Use ollama_chat as the preferred default
            litellm_params["model"] = f"ollama_chat/{model_id}"
            litellm_params["api_base"] = provider.base_url
    elif provider.type == "compat":
        # Compat models use model_id as-is
        litellm_params["model"] = model_id

    return litellm_params
