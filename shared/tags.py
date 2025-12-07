"""Tag utilities for providers and models."""
from __future__ import annotations

from typing import Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .models import ModelMetadata


def _normalize_tag(tag: str | None) -> str | None:
    """Normalize a tag string (trim, lowercase, hyphenate)."""
    if tag is None:
        return None
    value = str(tag).strip()
    if not value:
        return None

    # Replace internal whitespace with a single hyphen
    value = "-".join(value.replace(",", " ").split())
    return value.lower()


def normalize_tags(tags: Iterable[str] | None) -> list[str]:
    """Normalize and dedupe an iterable of tags."""
    seen = set()
    normalized: list[str] = []

    for tag in tags or []:
        clean = _normalize_tag(tag)
        if not clean or clean in seen:
            continue
        normalized.append(clean)
        seen.add(clean)

    return normalized


def parse_tags_input(raw: str | None) -> list[str]:
    """Parse comma- or newline-separated tags from user input."""
    if not raw:
        return []

    candidates: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        stripped = chunk.strip()
        if stripped:
            candidates.append(stripped)

    return normalize_tags(candidates)


def generate_model_tags(
    provider_name: str,
    provider_type: str,
    metadata: "ModelMetadata",
    provider_tags: Sequence[str] | None = None,
    mode: str | None = None,
) -> list[str]:
    """Generate system tags for a model using provider and metadata details."""
    tags: list[str] = [
        "lupdater",
        f"provider:{provider_name}",
        f"type:{provider_type}",
        f"model:{metadata.id}",
        f"unique_id:{provider_name}/{metadata.id}",
    ]

    # Only expose mode tag for Ollama providers; avoid leaking it on OpenAI/compat sources
    if mode and provider_type == "ollama":
        tags.append(f"mode:{mode}")

    if metadata.model_type:
        tags.append(f"model_type:{metadata.model_type}")

    for capability in metadata.capabilities or []:
        tags.append(f"capability:{capability}")

    tags.extend(metadata.tags or [])

    if provider_tags:
        tags.extend(provider_tags)

    return normalize_tags(tags)
