"""Pricing profile helpers for applying OpenAI-based defaults."""

from __future__ import annotations

from typing import Any

# Common OpenAI pricing profiles (tokens in USD)
OPENAI_PRICING_PROFILES: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input_cost_per_token": 0.000005,   # $5 per 1M
        "output_cost_per_token": 0.000015,  # $15 per 1M
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 0.000000150,  # $0.15 per 1M
        "output_cost_per_token": 0.000000600,  # $0.60 per 1M
    },
    "gpt-4.1": {
        "input_cost_per_token": 0.000015,  # $15 per 1M
        "output_cost_per_token": 0.000060,  # $60 per 1M
    },
    "o1-mini": {
        "input_cost_per_token": 0.000010,  # $10 per 1M
        "output_cost_per_token": 0.000040,  # $40 per 1M
    },
}


def _coerce_pricing_dict(raw: dict[str, Any] | None) -> dict[str, float]:
    """Return numeric pricing fields only."""
    if not isinstance(raw, dict):
        return {}
    pricing: dict[str, float] = {}
    for key in (
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_second",
        "output_cost_per_second",
        "output_cost_per_image",
        "input_cost_per_character",
        "output_cost_per_character",
    ):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            pricing[key] = float(value)
    return pricing


def build_pricing_from_profile(profile: str | None, override: dict[str, Any] | None = None) -> dict[str, float]:
    """Merge pricing from a known profile plus optional override."""
    pricing: dict[str, float] = {}
    if profile and profile in OPENAI_PRICING_PROFILES:
        pricing.update(OPENAI_PRICING_PROFILES[profile])
    pricing.update(_coerce_pricing_dict(override))
    return pricing


def apply_pricing_overrides(
    litellm_fields: dict[str, Any],
    *,
    config=None,
    provider=None,
    model=None,
) -> dict[str, Any]:
    """
    Apply pricing overrides with precedence: model > provider > global config.

    This preserves other litellm_fields keys.
    """
    merged = dict(litellm_fields)

    # Resolve profile/override precedence
    profile = None
    override = None

    if model is not None:
        profile = getattr(model, "pricing_profile", None)
        override = getattr(model, "pricing_override_dict", None)

    if provider is not None and profile is None:
        profile = getattr(provider, "pricing_profile", None)
    if provider is not None and override is None:
        override = getattr(provider, "pricing_override_dict", None)

    if config is not None and profile is None:
        profile = getattr(config, "default_pricing_profile", None)
    if config is not None and override is None:
        override = getattr(config, "default_pricing_override_dict", None)

    pricing = build_pricing_from_profile(profile, override)
    for key, value in pricing.items():
        merged[key] = value

    return merged
