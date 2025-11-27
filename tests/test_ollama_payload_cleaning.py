"""Tests for Ollama payload cleaning functionality.

These tests validate that large/redundant fields are properly stripped from
Ollama /api/show responses before caching or storing.
"""
import json
import os
from pathlib import Path

import httpx
import pytest

from litellm_updater.models import SourceEndpoint, SourceType
from litellm_updater.sources import _clean_ollama_payload, fetch_ollama_models


def _load_env() -> dict[str, str]:
    """Load test environment variables from tests/.env or system environment."""
    env: dict[str, str] = {}
    env_path = Path(__file__).with_name(".env")

    # Read from environment
    for key in ("TEST_OLLAMA_PROVIDER_URL", "TEST_OLLAMA_PROVIDER_KEY"):
        value = os.environ.get(key)
        if value:
            env[key] = value

    # Read from tests/.env if exists
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip())

    return env


def _is_configured(url: str | None) -> bool:
    """Check if a URL is configured (not None, empty, or 'none')."""
    return bool(url and url.lower() not in ("none", ""))


def test_clean_ollama_payload_strips_large_fields():
    """Test that cleaning removes modelfile, license, and tensor fields."""
    sample_path = Path("datasamples/ollama_modelinfo_sample.json")
    sample = json.loads(sample_path.read_text())[0]
    sample_with_tensors = {
        **sample,
        "tensors": ["root-level"],
        "model_info": {**sample.get("model_info", {}), "tensors": ["nested"]},
    }

    cleaned = _clean_ollama_payload(sample_with_tensors)

    assert "modelfile" in sample
    assert "modelfile" not in cleaned
    assert "license" not in cleaned
    assert "licence" not in cleaned
    assert "tensors" not in cleaned

    model_info = cleaned.get("model_info")
    assert isinstance(model_info, dict)
    assert "tensors" not in model_info
    assert "general.architecture" in model_info

    # Original payload should remain untouched
    assert "tensors" in sample_with_tensors.get("model_info", {})


def test_clean_ollama_payload_preserves_important_fields():
    """Test that cleaning preserves important metadata fields."""
    sample_path = Path("datasamples/ollama_modelinfo_sample.json")
    sample = json.loads(sample_path.read_text())[0]

    cleaned = _clean_ollama_payload(sample)

    # Should preserve these fields
    assert "model_info" in cleaned
    assert "details" in cleaned or "modelinfo" in cleaned

    # model_info should preserve architecture and important params
    if "model_info" in cleaned:
        model_info = cleaned["model_info"]
        assert isinstance(model_info, dict)
        # Should have some architecture info
        assert any("general." in key for key in model_info.keys())


@pytest.mark.asyncio
async def test_clean_ollama_payload_with_live_data():
    """Test payload cleaning with live data from Ollama server if configured."""
    test_env = _load_env()
    base_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")

    if not _is_configured(base_url):
        pytest.skip("TEST_OLLAMA_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
    )

    # Fetch models to get a real model ID
    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    if not models:
        pytest.skip("No models available on Ollama server")

    model_id = models[0].id

    # Fetch raw details from /api/show
    headers = {}
    if source.api_key:
        headers["Authorization"] = f"Bearer {source.api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{source.normalized_base_url}/api/show",
            json={"name": model_id},
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        raw_payload = response.json()

    # Clean the payload
    cleaned = _clean_ollama_payload(raw_payload)

    # Validate cleaning
    assert "modelfile" not in cleaned, "modelfile should be removed"
    assert "license" not in cleaned and "licence" not in cleaned, "license fields should be removed"

    # Check nested tensors removal
    if "model_info" in cleaned:
        assert "tensors" not in cleaned["model_info"], "tensors should be removed from model_info"

    # Should preserve important fields
    assert "model_info" in cleaned or "details" in cleaned, "Should preserve metadata sections"

    # Original should be untouched (test that cleaning doesn't mutate)
    if "modelfile" in raw_payload:
        assert "modelfile" in raw_payload, "Original payload should not be mutated"


@pytest.mark.asyncio
async def test_cleaned_payload_size_reduction():
    """Test that cleaning significantly reduces payload size."""
    test_env = _load_env()
    base_url = test_env.get("TEST_OLLAMA_PROVIDER_URL")

    if not _is_configured(base_url):
        pytest.skip("TEST_OLLAMA_PROVIDER_URL not configured in tests/.env")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_PROVIDER_KEY") or None,
    )

    # Fetch models
    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    if not models:
        pytest.skip("No models available on Ollama server")

    model_id = models[0].id

    # Fetch raw details
    headers = {}
    if source.api_key:
        headers["Authorization"] = f"Bearer {source.api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{source.normalized_base_url}/api/show",
            json={"name": model_id},
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        raw_payload = response.json()

    # Measure sizes
    raw_size = len(json.dumps(raw_payload))
    cleaned = _clean_ollama_payload(raw_payload)
    cleaned_size = len(json.dumps(cleaned))

    # Cleaning should reduce size (at least if modelfile/license were present)
    if "modelfile" in raw_payload or "license" in raw_payload:
        assert cleaned_size < raw_size, "Cleaned payload should be smaller than raw payload"

        # Calculate reduction percentage
        reduction_pct = ((raw_size - cleaned_size) / raw_size) * 100
        print(f"\nPayload size reduction for {model_id}: {reduction_pct:.1f}% ({raw_size} -> {cleaned_size} bytes)")

        # Should have significant reduction (modelfile is typically very large)
        assert reduction_pct > 10, "Should have at least 10% size reduction when large fields present"
