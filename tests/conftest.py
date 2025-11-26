"""Pytest helpers for integration checks."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Ensure the repository root is on sys.path so imports work without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_test_env() -> None:
    """Load missing TEST_* vars from tests/.env without overriding runtime env."""

    required_keys = {
        "TEST_OLLAMA_URL",
        "TEST_OPENAI_URL",
        "TEST_OPENAI_KEY",
    }
    optional_keys = {"TEST_OLLAMA_KEY"}

    missing = {key for key in required_keys if key not in os.environ}
    missing_optional = {key for key in optional_keys if key not in os.environ}
    if not missing and not missing_optional:
        return

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in missing or key in missing_optional:
            os.environ.setdefault(key.strip(), value.strip())


_load_test_env()

try:
    import pytest_asyncio as _pytest_asyncio  # type: ignore # noqa: F401

    HAS_PYTEST_ASYNCIO = True
except ImportError:
    HAS_PYTEST_ASYNCIO = False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "asyncio: async test requiring pytest-asyncio")
    if not HAS_PYTEST_ASYNCIO:
        config.addinivalue_line(
            "markers",
            "live: integration tests that hit configured upstream servers; skipped when pytest-asyncio is missing",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_PYTEST_ASYNCIO:
        return

    skip_asyncio = pytest.mark.skip(reason="Install dev extras (pip install -e .[dev]) to run asyncio tests")
    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(skip_asyncio)
