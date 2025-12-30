"""LiteLLM API client for pushing models."""
import logging
import httpx

from shared.models import ModelMetadata
from shared.pricing_profiles import apply_pricing_overrides
from shared.sources import _make_auth_headers, DEFAULT_TIMEOUT
from shared.tags import generate_model_tags, normalize_tags

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
                        model,
                        config=config,
                        session=session,
                    )
                    stats["added"] += 1
                    logger.info("Added model %s to LiteLLM", model.model_id)
                else:
                    # Model exists, check if update needed
                    litellm_model = litellm_index[unique_id_tag]
                    if await _needs_update(provider, model, litellm_model, config=config, session=session):
                        await update_model_in_litellm(
                            client,
                            config.litellm_base_url,
                            config.litellm_api_key,
                            provider,
                            model,
                            litellm_model,
                            config=config,
                            session=session,
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


def _collect_litellm_tags(model: dict) -> list[str]:
    """Collect tags across LiteLLM payload fields."""
    tags = model.get("litellm_params", {}).get("tags", [])
    model_info_tags = model.get("model_info", {}).get("tags", [])
    root_tags = model.get("tags", [])
    combined = list(tags or []) + list(model_info_tags or []) + list(root_tags or [])
    return [str(tag).lower() for tag in combined if tag is not None]


def _extract_tag_value(tags: list[str], prefix: str) -> str | None:
    """Extract the first tag value for a prefix."""
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return None


async def list_routing_group_deployments(config) -> list[dict]:
    """Return LiteLLM models tagged as routing groups."""
    if not config.litellm_base_url:
        return []

    entries: list[dict] = []
    async with httpx.AsyncClient() as client:
        litellm_models = await fetch_litellm_models(
            client,
            config.litellm_base_url,
            config.litellm_api_key,
        )

    for model in litellm_models:
        tags = _collect_litellm_tags(model)
        group_tag = next((tag for tag in tags if tag.startswith("routing_group:")), None)
        if not group_tag:
            continue
        group_name = group_tag.split(":", 1)[1]
        entries.append(
            {
                "group": group_name,
                "provider": _extract_tag_value(tags, "provider:") or "",
                "model_id": _extract_tag_value(tags, "model:") or "",
                "model_name": model.get("model_name"),
                "model_info_id": model.get("model_info", {}).get("id"),
                "created_by": model.get("model_info", {}).get("created_by"),
                "tags": tags,
            }
        )

    return entries


async def push_model_to_litellm(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    provider,
    model,
    config=None,
    session=None,
    model_name_override: str | None = None,
    extra_tags: list[str] | None = None,
    created_by: str = "updater",
    strip_unique_id: bool = False,
):
    """Push a single model to LiteLLM."""
    # Build litellm_params
    litellm_params = await _build_litellm_params(provider, model, session)

    # Build model_info with pricing overrides
    if provider.type == "compat" and session and model.mapped_provider_id and model.mapped_model_id:
        from shared.crud import get_provider_by_id, get_model_by_provider_and_name

        mapped_provider = await get_provider_by_id(session, model.mapped_provider_id)
        mapped_model = await get_model_by_provider_and_name(
            session, model.mapped_provider_id, model.mapped_model_id
        )
        if mapped_provider and mapped_model:
            model_info = apply_pricing_overrides(
                mapped_model.effective_params.copy(),
                config=config,
                provider=mapped_provider,
                model=mapped_model,
            )
            compat_overrides = model.effective_params.copy()
            compat_overrides.pop("tags", None)
            compat_overrides.pop("mode", None)
            model_info.update(compat_overrides)
        else:
            model_info = apply_pricing_overrides(
                model.effective_params.copy(),
                config=config,
                provider=provider,
                model=model,
            )
    else:
        model_info = apply_pricing_overrides(
            model.effective_params.copy(),
            config=config,
            provider=provider,
            model=model,
        )

    # Ensure default pricing fields are present if missing
    for key, value in model.litellm_params_dict.items():
        if "cost" in key or key == "tiered_pricing":
            model_info.setdefault(key, value)

    # Copy pricing fields into litellm_params so LiteLLM can bill requests
    _merge_pricing_fields(litellm_params, model_info)

    # Set litellm_provider
    ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama"

    # Auto-detect FIM (Fill-in-the-Middle) capability and add to model_info
    # Check if provider has auto_detect_fim enabled and model has FIM-related capabilities
    if provider.auto_detect_fim:
        effective_params = model.effective_params or {}
        if effective_params.get("supports_fill_in_middle") or effective_params.get("supports_code_infilling"):
            model_info["supports_fill_in_middle"] = True
            model_info["supports_code_infilling"] = True

    model_mode = _get_model_mode(model)

    if provider.type == "openai":
        model_info["litellm_provider"] = "openai"
        if model_mode == "completion":
            model_info["supports_completion"] = True
    elif provider.type == "ollama":
        model_info["mode"] = ollama_mode
        if ollama_mode == "openai":
            model_info["litellm_provider"] = "openai"
        else:
            model_info["litellm_provider"] = "ollama"
        if model_mode == "completion":
            model_info["supports_completion"] = True
    elif provider.type == "compat":
        compat_mode = _get_compat_mode(model)
        model_info.setdefault("mode", "completion" if compat_mode == "completion" else "chat")
        if compat_mode == "completion":
            model_info["supports_completion"] = True
        compat_model = litellm_params.get("model", "")
        if compat_model.startswith(("ollama/", "ollama_chat/")):
            model_info["litellm_provider"] = "ollama"
        elif compat_model:
            model_info["litellm_provider"] = "openai"

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
    if provider.type == "compat":
        compat_mode = _get_compat_mode(model)
        tags = [t for t in tags if not t.startswith("mode:")]
        if compat_mode == "completion":
            tags = [t for t in tags if t not in {"capability:chat", "capability:completion"}]
            tags.extend(["capability:completion", "mode:completion"])
        else:
            tags = [t for t in tags if t != "capability:completion"]
            tags.append("mode:chat")

    if strip_unique_id:
        tags = [t for t in tags if not t.startswith("unique_id:")]

    if extra_tags:
        tags.extend(normalize_tags(extra_tags))
        tags = normalize_tags(tags)

    litellm_params["tags"] = tags
    model_info["tags"] = tags

    # Add access_groups
    access_groups = model.get_effective_access_groups()
    if access_groups:
        model_info["access_groups"] = access_groups

    # Mark as created/updated by updater with timestamp
    from datetime import datetime, UTC
    current_time = datetime.now(UTC)
    model_info["created_by"] = created_by
    model_info["updated_at"] = current_time.isoformat()

    # Build display name
    display_name = model_name_override or model.get_display_name(apply_prefix=True)

    # Push to LiteLLM
    url = f"{base_url.rstrip('/')}/model/new"
    headers = _make_auth_headers(api_key)

    payload = {
        "model_name": display_name,
        "litellm_params": litellm_params,
        "model_info": model_info
    }

    # Debug logging for API key
    if litellm_params.get("api_key"):
        logger.info(f"Pushing model {display_name} with API key: {litellm_params['api_key'][:4]}***")
    else:
        logger.warning(f"Pushing model {display_name} WITHOUT API key! litellm_params: {litellm_params}")

    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


async def update_model_in_litellm(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    provider,
    model,
    litellm_model,
    config=None,
    session=None,
):
    """Update an existing model in LiteLLM."""
    # First delete the old model
    model_id = litellm_model.get('model_info', {}).get('id')
    if not model_id:
        logger.warning("Cannot update model without id, skipping")
        return

    await delete_model_from_litellm(client, base_url, api_key, model_id)

    # Then create the new version
    await push_model_to_litellm(client, base_url, api_key, provider, model, config=config, session=session)


async def delete_model_from_litellm(client: httpx.AsyncClient, base_url: str, api_key: str | None, model_id: str):
    """Delete a model from LiteLLM."""
    url = f"{base_url.rstrip('/')}/model/delete"
    headers = _make_auth_headers(api_key)

    payload = {"id": model_id}

    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


def _normalize_value(value):
    """Normalize a value for comparison (handles string/float pricing values)."""
    if value is None:
        return None
    # Try to convert strings to float for numeric comparison
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


async def _needs_update(provider, model, litellm_model, config=None, session=None) -> bool:
    """
    Check if a model in LiteLLM needs to be updated based on database model.

    Compares key fields to determine if re-sync is needed.
    """
    # Get current values from database model
    db_display_name = model.get_display_name(apply_prefix=True)
    db_params = await _build_litellm_params(provider, model, session)
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

    # Check pricing differences (with normalization to handle string/float inconsistencies)
    effective_info = apply_pricing_overrides(
        model.effective_params,
        config=config,
        provider=provider,
        model=model,
    )
    ll_info = litellm_model.get('model_info', {})
    ll_params = litellm_model.get('litellm_params', {})

    for key, value in effective_info.items():
        if "cost" not in key:
            continue
        ll_info_val = ll_info.get(key)
        ll_params_val = ll_params.get(key)

        # Normalize values for comparison (handles string/float pricing)
        db_normalized = _normalize_value(value)
        ll_info_normalized = _normalize_value(ll_info_val)
        ll_params_normalized = _normalize_value(ll_params_val)

        if ll_info_normalized != db_normalized or ll_params_normalized != db_normalized:
            logger.debug("Pricing changed for %s: ll_info=%s, ll_params=%s, db=%s", key, ll_info_val, ll_params_val, value)
            return True

    return False


def _get_compat_mode(model) -> str:
    """Return compat mode (chat or completion)."""
    mode = (model.user_params_dict or {}).get("mode")
    return "completion" if mode == "completion" else "chat"


def _get_model_mode(model) -> str:
    """Return model mode (chat or completion)."""
    mode = (model.user_params_dict or {}).get("mode")
    return "completion" if mode == "completion" else "chat"


async def _build_litellm_params(provider, model, session=None) -> dict:
    """Build litellm_params for a model."""
    litellm_params = {}
    model_id = model.model_id

    ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama_chat"
    request_mode = _get_model_mode(model)

    if provider.type == "openai":
        if request_mode == "completion":
            litellm_params["model"] = f"text-completion-openai/{model_id}"
        else:
            litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_base"] = provider.base_url
        litellm_params["api_key"] = provider.api_key or "sk-1234"
    elif provider.type == "ollama":
        if request_mode == "completion":
            if ollama_mode == "openai":
                litellm_params["model"] = f"text-completion-openai/{model_id}"
                api_base = provider.base_url.rstrip("/")
                litellm_params["api_base"] = f"{api_base}/v1"
                litellm_params["api_key"] = provider.api_key or "sk-1234"
            else:
                litellm_params["model"] = f"ollama/{model_id}"
                litellm_params["api_base"] = provider.base_url
                if provider.api_key:
                    litellm_params["api_key"] = provider.api_key
        elif ollama_mode == "openai":
            litellm_params["model"] = f"openai/{model_id}"
            api_base = provider.base_url.rstrip("/")
            litellm_params["api_base"] = f"{api_base}/v1"
            litellm_params["api_key"] = provider.api_key or "sk-1234"
        elif ollama_mode == "ollama":
            litellm_params["model"] = f"ollama/{model_id}"
            litellm_params["api_base"] = provider.base_url
            if provider.api_key:
                litellm_params["api_key"] = provider.api_key
        else:
            # Use ollama_chat as the preferred default
            litellm_params["model"] = f"ollama_chat/{model_id}"
            litellm_params["api_base"] = provider.base_url
            if provider.api_key:
                litellm_params["api_key"] = provider.api_key
    elif provider.type == "compat":
        # Compat models need to resolve their mapping to the actual provider/model
        compat_mode = _get_compat_mode(model)
        if compat_mode == "completion":
            litellm_params["supports_completion"] = True
        if not session:
            logger.warning("No session provided for compat model, using model_id as-is")
            litellm_params["model"] = model_id
        else:
            # Get the mapped provider and model
            from shared.crud import get_provider_by_id, get_model_by_provider_and_name

            if not model.mapped_provider_id or not model.mapped_model_id:
                logger.warning(f"Compat model {model_id} missing mapping, using model_id as-is")
                litellm_params["model"] = model_id
            else:
                mapped_provider = await get_provider_by_id(session, model.mapped_provider_id)
                if not mapped_provider:
                    logger.warning(f"Mapped provider {model.mapped_provider_id} not found for compat model {model_id}")
                    litellm_params["model"] = model_id
                else:
                    # Get the mapped model to determine ollama_mode
                    mapped_model = await get_model_by_provider_and_name(session, model.mapped_provider_id, model.mapped_model_id)
                    mapped_ollama_mode = (
                        model.ollama_mode or
                        (mapped_model.ollama_mode if mapped_model else None) or
                        mapped_provider.default_ollama_mode or
                        "ollama_chat"
                    )

                    # Build params based on mapped provider type
                    if mapped_provider.type == "openai":
                        if compat_mode == "completion":
                            litellm_params["model"] = f"text-completion-openai/{model.mapped_model_id}"
                        else:
                            litellm_params["model"] = f"openai/{model.mapped_model_id}"
                        litellm_params["api_base"] = mapped_provider.base_url
                        litellm_params["api_key"] = mapped_provider.api_key or "sk-1234"
                    elif mapped_provider.type == "ollama":
                        if mapped_ollama_mode == "openai":
                            if compat_mode == "completion":
                                litellm_params["model"] = f"text-completion-openai/{model.mapped_model_id}"
                            else:
                                litellm_params["model"] = f"openai/{model.mapped_model_id}"
                            api_base = mapped_provider.base_url.rstrip("/")
                            litellm_params["api_base"] = f"{api_base}/v1"
                            litellm_params["api_key"] = mapped_provider.api_key or "sk-1234"
                        elif mapped_ollama_mode == "ollama":
                            litellm_params["model"] = f"ollama/{model.mapped_model_id}"
                            litellm_params["api_base"] = mapped_provider.base_url
                            if mapped_provider.api_key:
                                litellm_params["api_key"] = mapped_provider.api_key
                        else:
                            # Use ollama_chat as the preferred default for chat mode
                            if compat_mode == "completion":
                                litellm_params["model"] = f"ollama/{model.mapped_model_id}"
                            else:
                                litellm_params["model"] = f"ollama_chat/{model.mapped_model_id}"
                            litellm_params["api_base"] = mapped_provider.base_url
                            if mapped_provider.api_key:
                                litellm_params["api_key"] = mapped_provider.api_key
                    else:
                        logger.warning(f"Unsupported mapped provider type {mapped_provider.type} for compat model {model_id}")
                        litellm_params["model"] = model_id

    return litellm_params


def _merge_pricing_fields(target: dict, source: dict) -> None:
    """Copy pricing-related fields from source into target."""
    for key, value in source.items():
        if value is None:
            continue
        if "cost" in key or key == "tiered_pricing":
            target[key] = value


async def push_routing_groups_to_litellm(session, config, group_id: int | None = None) -> dict:
    """Push routing groups to LiteLLM as model groups."""
    if not config.litellm_base_url:
        raise RuntimeError("LiteLLM destination not configured")

    from shared.crud import get_routing_groups, get_routing_group, get_model_by_provider_and_name, get_provider_by_id

    if group_id is None:
        groups = await get_routing_groups(session)
        groups = [await get_routing_group(session, g.id) for g in groups]
    else:
        group = await get_routing_group(session, group_id)
        groups = [group] if group else []

    groups = [g for g in groups if g is not None]
    stats = {"groups": len(groups), "added": 0, "deleted": 0, "missing_models": 0, "errors": 0}

    async with httpx.AsyncClient() as client:
        litellm_models = await fetch_litellm_models(client, config.litellm_base_url, config.litellm_api_key)

        for group in groups:
            group_tag = f"routing_group:{group.name}"
            group_tag_lower = group_tag.lower()

            for m in litellm_models:
                tags = m.get("litellm_params", {}).get("tags", [])
                model_info_tags = m.get("model_info", {}).get("tags", [])
                root_tags = m.get("tags", [])
                combined_tags = [str(t).lower() for t in (tags or []) + (model_info_tags or []) + (root_tags or [])]
                if group_tag_lower not in combined_tags:
                    continue
                if m.get("model_info", {}).get("created_by") != "routing_group":
                    continue
                model_id = m.get("model_info", {}).get("id")
                if not model_id:
                    continue
                try:
                    await delete_model_from_litellm(
                        client,
                        config.litellm_base_url,
                        config.litellm_api_key,
                        model_id,
                    )
                    stats["deleted"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    logger.warning("Failed deleting routing group entry %s: %s", model_id, exc)

            for target in sorted(group.targets, key=lambda t: (t.priority, t.id)):
                provider = target.provider or await get_provider_by_id(session, target.provider_id)
                if not provider:
                    stats["missing_models"] += 1
                    continue
                model = await get_model_by_provider_and_name(session, provider.id, target.model_id)
                if not model:
                    stats["missing_models"] += 1
                    continue
                try:
                    await push_model_to_litellm(
                        client,
                        config.litellm_base_url,
                        config.litellm_api_key,
                        provider,
                        model,
                        config=config,
                        session=session,
                        model_name_override=group.name,
                        extra_tags=[group_tag],
                        created_by="routing_group",
                        strip_unique_id=True,
                    )
                    stats["added"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    logger.warning(
                        "Failed pushing routing target %s/%s for group %s: %s",
                        provider.name,
                        model.model_id,
                        group.name,
                        exc,
                    )

    return stats
