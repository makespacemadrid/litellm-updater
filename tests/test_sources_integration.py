import os
from pathlib import Path

import httpx
import pytest

from litellm_updater.models import SourceEndpoint, SourceType
from litellm_updater.sources import fetch_litellm_models, fetch_ollama_models

REQUIRED_ENV_KEYS = (
    "TEST_OLLAMA_URL",
    "TEST_OPENAI_URL",
    "TEST_OPENAI_KEY",
)

OPTIONAL_ENV_KEYS = ("TEST_OLLAMA_KEY",)


def _load_env() -> dict[str, str]:
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


@pytest.fixture(scope="session")
def test_env() -> dict[str, str]:
    return _load_env()


@pytest.mark.asyncio
async def test_fetch_litellm_models_includes_metadata(test_env: dict[str, str]):
    base_url = test_env.get("TEST_OPENAI_URL")
    if not base_url:
        pytest.skip("TEST_OPENAI_URL is not configured in tests/.env or environment variables")

    source = SourceEndpoint(
        name="Test LiteLLM",
        base_url=base_url,
        type=SourceType.LITELLM,
        api_key=test_env.get("TEST_OPENAI_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_litellm_models(client, source)

    assert models, "Expected at least one model from the LiteLLM/OpenAI endpoint"
    metadata = models[0]

    assert metadata.id, "Model id should be populated"
    assert metadata.model_type, "Model type should be provided (e.g., completion or embedding)"
    assert metadata.capabilities, "Capabilities should include tool/vision/chat support where available"
    assert metadata.context_window is not None, "Context size should be parsed from model metadata"


@pytest.mark.asyncio
async def test_fetch_ollama_models_uses_show_endpoint(test_env: dict[str, str]):
    base_url = test_env.get("TEST_OLLAMA_URL")
    if not base_url:
        pytest.skip("TEST_OLLAMA_URL is not configured in tests/.env or environment variables")

    source = SourceEndpoint(
        name="Test Ollama",
        base_url=base_url,
        type=SourceType.OLLAMA,
        api_key=test_env.get("TEST_OLLAMA_KEY") or None,
    )

    async with httpx.AsyncClient() as client:
        models = await fetch_ollama_models(client, source)

    assert models, "Expected at least one model from the Ollama server"
    metadata = models[0]

    assert metadata.id, "Model id should be populated"
    assert metadata.capabilities, "Capabilities should be pulled from the /api/show response"
    assert metadata.model_type, "Model type should be inferred from capabilities or upstream metadata"
    assert metadata.context_window is not None, "Context size should be parsed from the Ollama model_info"
