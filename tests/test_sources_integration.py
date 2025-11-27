"""Integration tests for source fetchers using live provider endpoints.

These tests connect to real Ollama/OpenAI/LiteLLM servers configured via tests/.env
and validate that model metadata is correctly parsed and mapped to LiteLLM fields.
"""
import os
from pathlib import Path

import httpx
import pytest

from litellm_updater.models import (
    LITELLM_MODEL_FIELDS,
    ModelMetadata,
    SourceEndpoint,
    SourceType,
)
from litellm_updater.sources import (
    fetch_litellm_models,
    fetch_ollama_model_details,
    fetch_ollama_models,
)

REQUIRED_ENV_KEYS = (
    "TEST_OLLAMA_PROVIDER_URL",
    "TEST_OPENAI_PROVIDER_URL",
    "TEST_OPENAI_PROVIDER_KEY",
)

OPTIONAL_ENV_KEYS = ("TEST_OLLAMA_PROVIDER_KEY",)


def _load_env() -> dict[str, str]:
    """Load test environment variables from tests/.env or system environment."""
    env: dict[str, str] = {}
    env_path = Path(__file__).with_name(".env")
    file_vars: dict[str, str] = {}

    # Prioritize values already exported in the environment
    for key in (*REQUIRED_ENV_KEYS, *OPTIONAL_ENV_KEYS):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    # Only read from tests/.env when a required value is missing
    missing_required = [key for key in REQUIRED_ENV_KEYS if key not in env]
    missing_optional = [key for key in OPTIONAL_ENV_KEYS if key not in env]
    if (missing_required or missing_optional) and env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            file_vars[key.strip()] = value.strip()

    for key in missing_required:
        if key in file_vars:
            env[key] = file_vars[key]

    for key in missing_optional:
        if key in file_vars:
            env[key] = file_vars[key]

    return env


def _is_configured(url: str | None) -> bool:
    """Check if a URL is configured (not None, empty, or 'none')."""
    return bool(url and url.lower() not in ("none", ""))


@pytest.fixture(scope="session")
def test_env() -> dict[str, str]:
    return _load_env()


# ==================== Ollama Tests ====================


@pytest.mark.asyncio
async def test_ollama_connectivity(test_env: dict[str, str]):
    """Test connectivity to Ollama server and fetch model list."""
    base_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OLLAMA_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    assert models, "Expected at least one model from Ollama server"
    assert all(isinstance(m, ModelMetadata) for m in models), "All items should be ModelMetadata"


@pytest.mark.asyncio
async def test_ollama_model_metadata_parsing(test_env: dict[str, str]):
    """Test that Ollama model metadata is correctly parsed from live data."""
    base_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OLLAMA_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    assert models, "Expected at least one model from Ollama"

    for model in models:
        # Validate basic fields
        assert model.id, f"Model ID should be populated"
        assert model.raw, f"Raw metadata should be stored for {model.id}"
        assert model.raw.get("name") == model.id, f"Raw name should match ID for {model.id}"

        # Validate expected Ollama response fields
        assert any(
            key in model.raw for key in ("digest", "size", "details", "modified_at")
        ), f"Expected Ollama /api/tags fields in {model.id}"


@pytest.mark.asyncio
async def test_ollama_litellm_field_mapping(test_env: dict[str, str]):
    """Test that all mappable fields from Ollama are correctly identified for LiteLLM."""
    base_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OLLAMA_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    assert models, "Expected at least one model from Ollama"

    for model in models:
        litellm_fields = model.litellm_fields

        # Validate required LiteLLM fields
        assert "litellm_provider" in litellm_fields, f"litellm_provider missing for {model.id}"
        assert litellm_fields["litellm_provider"] == "ollama", f"litellm_provider should be 'ollama' for {model.id}"

        # Validate pricing fields are present (defaults added)
        pricing_fields = [k for k in litellm_fields if "cost" in k]
        assert pricing_fields, f"At least one pricing field should be present for {model.id}"

        # All keys in litellm_fields should be in LITELLM_MODEL_FIELDS
        for key in litellm_fields:
            assert key in LITELLM_MODEL_FIELDS, (
                f"Field '{key}' in litellm_fields for {model.id} is not in LITELLM_MODEL_FIELDS"
            )

        # Validate context window if available
        if model.context_window:
            assert "max_input_tokens" in litellm_fields, f"max_input_tokens should be set when context_window exists for {model.id}"

        # Validate capabilities mapping
        if model.capabilities:
            supports_fields = [k for k in litellm_fields if k.startswith("supports_")]
            # At least some capabilities should map to supports_* fields
            if any(cap.lower() in ("vision", "tools", "function calling", "completion", "chat")
                   for cap in model.capabilities):
                assert supports_fields, f"Capabilities {model.capabilities} should map to supports_* fields for {model.id}"


@pytest.mark.asyncio
async def test_ollama_model_details_fetch(test_env: dict[str, str]):
    """Test fetching detailed Ollama model info via /api/show endpoint."""
    base_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OLLAMA_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    assert models, "Expected at least one model from Ollama"
    model_id = models[0].id

    # Fetch detailed info
    details = await fetch_ollama_model_details(source, model_id)

    assert isinstance(details, dict) and details, f"Show endpoint should return details for {model_id}"
    assert "modelfile" not in details, "Large 'modelfile' field should be cleaned from response"
    assert "license" not in details and "licence" not in details, "License fields should be cleaned"

    # Should have model_info section
    if "model_info" in details:
        assert isinstance(details["model_info"], dict), "model_info should be a dict"
        assert "tensors" not in details["model_info"], "Tensors should be cleaned from model_info"


# ==================== OpenAI/LiteLLM Tests ====================


@pytest.mark.asyncio
async def test_openai_connectivity(test_env: dict[str, str]):
    """Test connectivity to OpenAI/LiteLLM-compatible server and fetch model list."""
    base_url = test_env.get("TEST_OPENAI_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OPENAI_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test OpenAI",
        base_url=base_url,
        type=SourceType.LITELLM,
        api_key=test_env.get("TEST_OPENAI_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_litellm_models(client, source)

    assert models, "Expected at least one model from OpenAI/LiteLLM endpoint"
    assert all(isinstance(m, ModelMetadata) for m in models), "All items should be ModelMetadata"


@pytest.mark.asyncio
async def test_openai_model_metadata_parsing(test_env: dict[str, str]):
    """Test that OpenAI/LiteLLM model metadata is correctly parsed from live data."""
    base_url = test_env.get("TEST_OPENAI_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OPENAI_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test OpenAI",
        base_url=base_url,
        type=SourceType.LITELLM,
        api_key=test_env.get("TEST_OPENAI_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_litellm_models(client, source)

    assert models, "Expected at least one model from OpenAI/LiteLLM"

    for model in models:
        # Validate basic fields
        assert model.id, f"Model ID should be populated"
        assert model.raw, f"Raw metadata should be stored for {model.id}"

        # OpenAI models typically have 'object' field
        assert "object" in model.raw or "id" in model.raw, (
            f"Expected 'object' or 'id' field in OpenAI response for {model.id}"
        )


@pytest.mark.asyncio
async def test_openai_litellm_field_mapping(test_env: dict[str, str]):
    """Test that all mappable fields from OpenAI/LiteLLM are correctly identified."""
    base_url = test_env.get("TEST_OPENAI_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OPENAI_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test OpenAI",
        base_url=base_url,
        type=SourceType.LITELLM,
        api_key=test_env.get("TEST_OPENAI_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_litellm_models(client, source)

    assert models, "Expected at least one model from OpenAI/LiteLLM"

    for model in models:
        litellm_fields = model.litellm_fields

        # Validate that litellm_provider is set
        assert "litellm_provider" in litellm_fields, f"litellm_provider missing for {model.id}"

        # Validate pricing fields are present (defaults added)
        pricing_fields = [k for k in litellm_fields if "cost" in k]
        assert pricing_fields, f"At least one pricing field should be present for {model.id}"

        # All keys in litellm_fields should be in LITELLM_MODEL_FIELDS
        for key in litellm_fields:
            assert key in LITELLM_MODEL_FIELDS, (
                f"Field '{key}' in litellm_fields for {model.id} is not in LITELLM_MODEL_FIELDS"
            )

        # Validate context window mapping
        if model.context_window:
            assert "max_input_tokens" in litellm_fields, (
                f"max_input_tokens should be set when context_window exists for {model.id}"
            )

        # Validate model type detection
        if model.model_type:
            assert isinstance(model.model_type, str), f"model_type should be string for {model.id}"

        # Validate capabilities mapping
        if model.capabilities:
            # Check that capabilities are properly mapped to supports_* fields
            cap_lower = [c.lower() for c in model.capabilities]
            if "vision" in cap_lower:
                assert litellm_fields.get("supports_vision") is True, (
                    f"Vision capability should map to supports_vision for {model.id}"
                )
            if any(kw in cap_lower for kw in ("tools", "function calling", "function_calling")):
                assert litellm_fields.get("supports_function_calling") is True, (
                    f"Tools capability should map to supports_function_calling for {model.id}"
                )


@pytest.mark.asyncio
async def test_openai_detailed_metadata_extraction(test_env: dict[str, str]):
    """Test extraction of detailed metadata fields from OpenAI/LiteLLM responses."""
    base_url = test_env.get("TEST_OPENAI_PROVIDER_URL")
    if not _is_configured(base_url):
        pytest.skip("TEST_OPENAI_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test OpenAI",
        base_url=base_url,
        type=SourceType.LITELLM,
        api_key=test_env.get("TEST_OPENAI_PROVIDER_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_litellm_models(client, source)

    assert models, "Expected at least one model from OpenAI/LiteLLM"

    # Test at least one model for detailed field extraction
    model = models[0]
    litellm_fields = model.litellm_fields

    # Check that fields from nested sections are extracted
    # (metadata, details, model_info, summary)
    raw_sections = ["metadata", "details", "model_info", "summary"]
    found_fields_from_raw = []

    for section_name in raw_sections:
        section = model.raw.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                if key in LITELLM_MODEL_FIELDS and value not in (None, "", [], {}):
                    found_fields_from_raw.append(key)

    # All found fields should be in litellm_fields
    for field in found_fields_from_raw:
        assert field in litellm_fields, (
            f"Field '{field}' found in raw sections but not in litellm_fields for {model.id}"
        )


# ==================== Cross-Provider Tests ====================


@pytest.mark.asyncio
async def test_all_providers_litellm_field_consistency(test_env: dict[str, str]):
    """Test that all providers produce consistent LiteLLM field mappings."""
    all_models = []

    # Fetch from Ollama if configured
    ollama_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")
    if _is_configured(ollama_url):
        source = SourceEndpoint(
            name="Test Ollama",
            base_url=ollama_url,
            type=SourceType.OLLAMA,
            api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
        )
        async with httpx.AsyncClient() as client:
            ollama_models = await fetch_ollama_models(client, source)
            all_models.extend(ollama_models)

    # Fetch from OpenAI/LiteLLM if configured
    openai_url = test_env.get("TEST_OPENAI_PROVIDER_URL")
    if _is_configured(openai_url):
        source = SourceEndpoint(
            name="Test OpenAI",
            base_url=openai_url,
            type=SourceType.LITELLM,
            api_key=test_env.get("TEST_OPENAI_PROVIDER_KEY") or None,
        )
        async with httpx.AsyncClient() as client:
            openai_models = await fetch_litellm_models(client, source)
            all_models.extend(openai_models)

    if not all_models:
        pytest.skip("No providers configured in tests/.env")

    # All models should have consistent LiteLLM field structure
    for model in all_models:
        litellm_fields = model.litellm_fields

        # All should have litellm_provider
        assert "litellm_provider" in litellm_fields

        # All should have at least one pricing field
        assert any("cost" in k for k in litellm_fields)

        # All fields should be valid LiteLLM fields
        invalid_fields = [k for k in litellm_fields if k not in LITELLM_MODEL_FIELDS]
        assert not invalid_fields, (
            f"Invalid LiteLLM fields {invalid_fields} found for {model.id}"
        )

        # No None or empty values
        for key, value in litellm_fields.items():
            assert value not in (None, "", [], {}), (
                f"Field '{key}' has empty value for {model.id}"
            )
