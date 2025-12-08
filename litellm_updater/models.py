"""Pydantic models and enums used across the LiteLLM updater service."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, computed_field, model_validator

from .tags import normalize_tags


class SourceType(str, Enum):
    """Supported upstream source types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    COMPAT = "compat"

    def display_name(self) -> str:
        """Return human-readable name for UI display."""
        if self is SourceType.OLLAMA:
            return "Ollama"
        elif self is SourceType.OPENAI:
            return "OpenAI"
        else:
            return "Compat"


class SourceEndpoint(BaseModel):
    """Configuration for a single upstream source server."""

    name: str = Field(..., description="Display name for the source")
    base_url: HttpUrl = Field(..., description="Base URL for the upstream server")
    type: SourceType = Field(..., description="Type of the upstream server")
    api_key: str | None = Field(
        None,
        description="Optional API key or bearer token used to authenticate against the source",
    )
    prefix: str | None = Field(
        None,
        description="Optional prefix for model names (e.g., 'mks-ollama'). Applied to display names and model_name in LiteLLM",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags applied to all models from this source/provider",
    )
    default_ollama_mode: str | None = Field(
        None,
        description="Default Ollama mode: 'ollama' or 'openai'. Only valid for Ollama sources",
    )

    @model_validator(mode="after")
    def validate_ollama_mode(self) -> "SourceEndpoint":
        """Validate ollama_mode only for Ollama sources."""
        if self.default_ollama_mode is not None:
            if self.type != SourceType.OLLAMA:
                raise ValueError("default_ollama_mode is only valid for Ollama sources")
            if self.default_ollama_mode not in ("ollama", "ollama_chat", "openai"):
                raise ValueError("default_ollama_mode must be 'ollama', 'ollama_chat', or 'openai'")
        elif self.type == SourceType.OLLAMA:
            # Default to chat-friendly Ollama mode when not provided
            self.default_ollama_mode = "ollama_chat"
        else:
            self.default_ollama_mode = None
        return self

    @model_validator(mode="after")
    def normalize_tags(self) -> "SourceEndpoint":
        """Normalize tags for consistency."""
        self.tags = normalize_tags(self.tags)
        return self

    @property
    def normalized_base_url(self) -> str:
        """Return the base URL without a trailing slash for safe path joining."""
        return str(self.base_url).rstrip("/")

    def apply_prefix(self, model_name: str) -> str:
        """Apply prefix to model name if configured."""
        if self.prefix:
            return f"{self.prefix}/{model_name}"
        return model_name


class LitellmDestination(BaseModel):
    """Destination LiteLLM proxy configuration."""

    base_url: HttpUrl | None = Field(
        None, description="LiteLLM base URL. Leave empty to disable synchronization."
    )
    api_key: str | None = Field(None, description="API key to authenticate LiteLLM admin calls")

    @property
    def configured(self) -> bool:
        """Return True when a LiteLLM endpoint has been configured."""

        return self.base_url is not None

    @property
    def normalized_base_url(self) -> str:
        """Return the base URL without a trailing slash for safe path joining."""

        if self.base_url is None:
            raise ValueError("LiteLLM endpoint is not configured")
        return str(self.base_url).rstrip("/")


def _extract_numeric(raw: dict, *keys: str) -> int | None:
    """Return the first numeric value found for the given keys in common sections."""

    def _search(mapping: dict | None) -> int | None:
        if not isinstance(mapping, dict):
            return None

        for key in keys:
            value = mapping.get(key)
            if isinstance(value, (int, float)):
                return int(value)

        for key, value in mapping.items():
            if isinstance(value, (int, float)) and any(str(key).endswith(candidate) for candidate in keys):
                return int(value)

        return None

    # First try direct sections
    for section in (
        raw,
        raw.get("metadata"),
        raw.get("details"),
        raw.get("summary"),
    ):
        found = _search(section)
        if found is not None:
            return found

    # For Ollama, check model_info with architecture-specific keys
    model_info = raw.get("model_info")
    if isinstance(model_info, dict):
        # Try architecture-specific keys (e.g., "qwen3.context_length")
        for key, value in model_info.items():
            if isinstance(value, (int, float)) and any(str(key).endswith(f".{candidate}") for candidate in keys):
                return int(value)
        # Also try direct keys
        found = _search(model_info)
        if found is not None:
            return found

    return None


def _extract_text(raw: dict, *keys: str) -> str | None:
    """Return the first non-empty string value found for the given keys."""

    for section in (
        raw,
        raw.get("metadata"),
        raw.get("details"),
        raw.get("model_info"),
        raw.get("summary"),
    ):
        if not isinstance(section, dict):
            continue

        for key in keys:
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return None


def _dedupe(values: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""

    seen = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _extract_capabilities(raw: dict) -> list[str]:
    """Normalize capability-like fields into a readable list."""

    capabilities: list[str] = []

    raw_capabilities = raw.get("capabilities")
    if isinstance(raw_capabilities, dict):
        for name, enabled in raw_capabilities.items():
            if enabled:
                capabilities.append(str(name).replace("_", " "))
    elif isinstance(raw_capabilities, list):
        capabilities.extend(str(value) for value in raw_capabilities if value)

    for field in ("modalities", "tools"):
        values = raw.get(field)
        if isinstance(values, list):
            capabilities.extend(str(value) for value in values if value)

    # Infer capabilities from Ollama model details and families
    details = raw.get("details")
    if isinstance(details, dict):
        families = details.get("families", [])
        if isinstance(families, list):
            # Vision support - models with clip, llava, or vision in families
            if any(fam.lower() in ["clip", "llava", "vision", "bakllava", "moondream"] for fam in families):
                capabilities.append("vision")

            # Audio support - whisper models
            if any(fam.lower() in ["whisper"] for fam in families):
                capabilities.append("audio")

            # Check main family too
            family = details.get("family", "").lower()
            if family in ["clip", "llava", "bakllava", "moondream"]:
                capabilities.append("vision")
            if family == "whisper":
                capabilities.append("audio")

    return _dedupe(capabilities)


def _map_capabilities_to_supports(capabilities: list[str]) -> dict[str, Any]:
    """Map capability strings to LiteLLM supports_* boolean fields."""

    supports: dict[str, Any] = {}

    cap_lower = [str(c).lower() for c in capabilities]

    # Map capabilities to supports fields
    if "vision" in cap_lower:
        supports["supports_vision"] = True
    if "tools" in cap_lower or "function calling" in cap_lower or "function_calling" in cap_lower:
        supports["supports_function_calling"] = True
        supports["supports_tool_choice"] = True
    if "completion" in cap_lower or "chat" in cap_lower:
        supports["supports_system_messages"] = True
    if "thinking" in cap_lower or "reasoning" in cap_lower:
        supports["supports_reasoning"] = True
    if "embedding" in cap_lower:
        # Embedding models typically don't support streaming the same way
        pass
    if "audio" in cap_lower or "audio input" in cap_lower:
        supports["supports_audio_input"] = True
    if "audio output" in cap_lower:
        supports["supports_audio_output"] = True
    if "pdf" in cap_lower or "pdf input" in cap_lower:
        supports["supports_pdf_input"] = True
    if "web search" in cap_lower or "web_search" in cap_lower:
        supports["supports_web_search"] = True

    return supports


def _detect_ollama_mode(model_id: str, family: str, capabilities: list[str]) -> str:
    """Detect LiteLLM mode for Ollama models based on name and family."""
    name_lower = model_id.lower()
    family_lower = family.lower() if family else ""

    # Embedding models
    if any(x in name_lower for x in ["embed", "embedding"]):
        return "embedding"
    if any(x in family_lower for x in ["bert", "nomic-bert"]):
        return "embedding"

    # Vision models use chat mode in LiteLLM
    if any(x in name_lower for x in ["vision", "-vl", "vlm"]):
        return "chat"

    # Everything else defaults to chat
    return "chat"


def _extract_model_type(model_id: str, raw: dict, capabilities: list[str]) -> str | None:
    """Infer a model type such as embeddings or completion from known fields."""

    for key in ("type", "model_type", "task"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value

    details = raw.get("details")
    if isinstance(details, dict):
        # Use Ollama family to detect mode
        family = details.get("family", "")
        if family:
            mode = _detect_ollama_mode(model_id, family, capabilities)
            return mode

        for key in ("type", "model_type"):
            value = details.get(key)
            if isinstance(value, str) and value:
                return value

    if capabilities:
        return capabilities[0]

    normalized_id = model_id.lower()
    if "embed" in normalized_id:
        return "embedding"
    if any(token in normalized_id for token in ("gpt", "llama", "mixtral", "turbo", "chat")):
        return "completion"
    return None


def _ensure_capabilities(model_id: str, capabilities: list[str], model_type: str | None) -> list[str]:
    """Backfill capabilities using heuristics when upstream data is sparse."""

    normalized = list(capabilities)
    lowered_id = model_id.lower()

    # Infer vision capability from model name
    vision_keywords = ["vision", "llava", "bakllava", "moondream", "minicpm-v", "cogvlm", "qwen-vl", "qwen2-vl", "qwen3-vl"]
    if any(keyword in lowered_id for keyword in vision_keywords):
        if "vision" not in [c.lower() for c in normalized]:
            normalized.append("vision")

    # Infer function calling / tool use from model name
    tool_keywords = ["tool", "function", "groq-tool"]
    if any(keyword in lowered_id for keyword in tool_keywords):
        if "function calling" not in [c.lower() for c in normalized]:
            normalized.append("function calling")

    # Infer audio capability from model name
    audio_keywords = ["whisper", "audio"]
    if any(keyword in lowered_id for keyword in audio_keywords):
        if "audio" not in [c.lower() for c in normalized]:
            normalized.append("audio")

    # Infer embedding capability from model name or type
    embedding_keywords = ["embed", "embedding"]
    if (
        any(keyword in lowered_id for keyword in embedding_keywords)
        or (model_type and any(keyword in model_type.lower() for keyword in embedding_keywords))
    ):
        if "embedding" not in [c.lower() for c in normalized]:
            normalized.append("embedding")

    # If still no capabilities, default to chat for non-embedding models
    if not normalized:
        # Don't add chat/completion if it looks like an embedding model
        if not any(keyword in lowered_id for keyword in embedding_keywords):
            normalized.append("chat")

    return _dedupe(normalized)


def _fallback_context_window(model_id: str, context_window: int | None) -> int | None:
    """Infer a reasonable context window when upstream metadata omits it."""

    if context_window is not None:
        return context_window

    lowered_id = model_id.lower()

    # Try to extract from model name (e.g., "128k", "32k")
    match = re.search(r"(\d+)k", lowered_id)
    if match:
        try:
            return int(match.group(1)) * 1000
        except ValueError:
            pass

    # Common model families with known context windows
    if "gpt-3.5" in lowered_id:
        return 16385
    if "gpt-4" in lowered_id or "gpt4" in lowered_id:
        return 128000

    # Llama models
    if "llama3.1" in lowered_id or "llama-3.1" in lowered_id:
        return 128000
    if "llama3.2" in lowered_id or "llama-3.2" in lowered_id:
        return 128000
    if "llama3" in lowered_id or "llama-3" in lowered_id:
        return 8192
    if "llama2" in lowered_id or "llama-2" in lowered_id:
        return 4096

    # Qwen models
    if "qwen2.5" in lowered_id or "qwen-2.5" in lowered_id:
        return 32768
    if "qwen2" in lowered_id or "qwen-2" in lowered_id:
        return 32768
    if "qwen3" in lowered_id or "qwen-3" in lowered_id:
        return 32768

    # Mistral models
    if "mistral" in lowered_id or "mixtral" in lowered_id:
        if "large" in lowered_id:
            return 128000
        return 32000

    # Gemma models
    if "gemma2" in lowered_id or "gemma-2" in lowered_id:
        return 8192
    if "gemma" in lowered_id:
        return 8192

    # Claude models
    if "claude-3" in lowered_id:
        return 200000
    if "claude-2" in lowered_id:
        return 100000

    return None


def _extract_tags(raw: dict) -> list[str]:
    """Extract and normalize tags from common payload sections."""

    def _collect(section: dict | None) -> list[str]:
        if not isinstance(section, dict):
            return []
        tags_field = section.get("tags")
        if isinstance(tags_field, list):
            return [str(tag) for tag in tags_field if tag]
        if isinstance(tags_field, str):
            pieces = tags_field.replace(";", ",").split(",")
            return [piece.strip() for piece in pieces if piece.strip()]
        return []

    tags: list[str] = []
    for section in (raw, raw.get("metadata"), raw.get("details"), raw.get("summary"), raw.get("litellm_params"), raw.get("model_info")):
        tags.extend(_collect(section))

    general = raw.get("general")
    if isinstance(general, dict):
        tags.extend(_collect(general))

    return normalize_tags(tags)


def _get_default_pricing(model_type: str | None, mode: str | None) -> dict[str, Any]:
    """Get default pricing using OpenAI-style fields for chat, audio, and images.

    Pricing reference (2025):
    - GPT-4: $0.00003/token input, $0.00006/token output
    - Whisper: $0.0001/second (audio transcription)
    - TTS (gpt-4o-mini-tts): $0.000015/character input
    - DALL-E 3: $0.08/image (average)
    """

    pricing: dict[str, Any] = {}

    lower_mode = (mode or "").lower() if mode else None
    lower_type = (model_type or "").lower() if model_type else ""

    # Audio transcription (Whisper)
    if lower_mode == "audio_transcription" or "audio_transcription" in lower_type or "whisper" in lower_type:
        # Whisper pricing: $0.006 per minute = $0.0001 per second
        pricing["input_cost_per_second"] = 0.0001
        pricing["output_cost_per_second"] = 0.0
        return pricing

    # Text-to-speech (audio_speech)
    if lower_mode == "audio_speech" or "tts" in lower_type or "speech" in lower_type:
        # TTS pricing: $15 per 1M characters = $0.000015 per character
        pricing["input_cost_per_character"] = 0.000015
        return pricing

    # Determine if this is an image generation model
    if lower_type and any(
        keyword in lower_type for keyword in ["image", "dall-e", "dalle", "vision-gen"]
    ):
        # DALL-E 3 average pricing: $0.08 per image
        pricing["output_cost_per_image"] = 0.08
        return pricing

    # Default to GPT-4 chat/completion pricing
    # GPT-4: $30 per 1M input tokens, $60 per 1M output tokens
    pricing["input_cost_per_token"] = 0.00003  # $0.03 per 1K tokens
    pricing["output_cost_per_token"] = 0.00006  # $0.06 per 1K tokens

    return pricing


# Fields recognized by LiteLLM model definitions that are useful to surface in the UI
LITELLM_MODEL_FIELDS: set[str] = {
    "aliases",
    "annotation_cost_per_page",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost",
    "cache_read_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_flex",
    "cache_read_input_token_cost_priority",
    "citation_cost_per_token",
    "input_cost_per_audio_token",
    "input_cost_per_character",
    "input_cost_per_query",
    "input_cost_per_second",
    "input_cost_per_token",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_batches",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "key",
    "litellm_provider",
    "max_input_tokens",
    "max_output_tokens",
    "max_tokens",
    "mode",
    "ocr_cost_per_page",
    "output_cost_per_audio_token",
    "output_cost_per_character",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image",
    "output_cost_per_image_token",
    "output_cost_per_reasoning_token",
    "output_cost_per_second",
    "output_cost_per_token",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_video_per_second",
    "output_vector_size",
    "rpm",
    "search_context_cost_per_query",
    "source",
    "supported_openai_params",
    "supports_assistant_prefill",
    "supports_audio_input",
    "supports_audio_output",
    "supports_computer_use",
    "supports_embedding_image_input",
    "supports_function_calling",
    "supports_native_streaming",
    "supports_pdf_input",
    "supports_prompt_caching",
    "supports_reasoning",
    "supports_response_schema",
    "supports_system_messages",
    "supports_tool_choice",
    "supports_url_context",
    "supports_vision",
    "supports_web_search",
    "tags",
    "tiered_pricing",
    "tpm",
}


class ModelMetadata(BaseModel):
    """Normalized model description."""

    id: str
    database_id: str | None = Field(None, description="Database UUID for LiteLLM models (used for deletion)")
    model_type: str | None = Field(None, description="Model type such as embeddings or completion")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    max_tokens: int | None = Field(None, description="Combined token limit when provided")
    max_input_tokens: int | None = Field(None, description="Maximum input tokens accepted")
    context_window: int | None = Field(
        None, description="Maximum input context window (tokens) when provided by the source"
    )
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens supported when provided by the source"
    )
    litellm_mode: str | None = Field(None, description="LiteLLM mode such as chat or embeddings")
    capabilities: list[str] = Field(
        default_factory=list, description="Normalized list of model capabilities or modalities"
    )
    tags: list[str] = Field(default_factory=list, description="Tags extracted from the source payload")
    raw: dict = Field(default_factory=dict, description="Raw metadata returned from the source")

    @classmethod
    def from_raw(cls, model_id: str, raw: dict, database_id: str | None = None) -> "ModelMetadata":
        """Construct a metadata object with normalized context and capability details."""

        max_tokens = _extract_numeric(raw, "max_tokens")
        max_input_tokens = _extract_numeric(
            raw, "max_input_tokens", "context_length", "context_window", "max_context"
        )
        context_window = _extract_numeric(raw, "context_length", "context_window", "max_context")
        context_window = context_window or max_input_tokens
        max_output_tokens = _extract_numeric(raw, "max_output_tokens", "max_output_length")
        litellm_mode = _extract_text(raw, "mode")
        capabilities = _extract_capabilities(raw)
        model_type = _extract_model_type(model_id, raw, capabilities)
        capabilities = _ensure_capabilities(model_id, capabilities, model_type)
        context_window = _fallback_context_window(model_id, context_window)
        tags = _extract_tags(raw)

        return cls(
            id=model_id,
            database_id=database_id,
            model_type=model_type,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            litellm_mode=litellm_mode,
            capabilities=capabilities,
            tags=tags,
            raw=raw,
        )

    @computed_field
    @property
    def litellm_fields(self) -> dict[str, Any]:
        """Return LiteLLM-compatible fields from the raw payload, omitting nulls."""

        def _collect(section: dict | None) -> dict[str, Any]:
            if not isinstance(section, dict):
                return {}
            return {
                key: value
                for key, value in section.items()
                if key in LITELLM_MODEL_FIELDS and value not in (None, "", [], {})
            }

        merged: dict[str, Any] = {}
        for section in (
            self.raw.get("model_info"),
            self.raw.get("details"),
            self.raw.get("metadata"),
            self.raw.get("summary"),
            self.raw,
        ):
            for key, value in _collect(section).items():
                merged.setdefault(key, value)

        # Extract Ollama-specific metadata for tags
        ollama_tags = []
        details = self.raw.get("details")
        if isinstance(details, dict):
            family = details.get("family")
            param_size = details.get("parameter_size")
            quant = details.get("quantization_level")

            if family:
                ollama_tags.append(f"family:{family}")
            if param_size:
                ollama_tags.append(f"size:{param_size}")
            if quant:
                ollama_tags.append(f"quant:{quant}")

        # Combine with source tags
        source_tags = normalize_tags(self.tags + ollama_tags)
        if source_tags:
            merged["tags"] = source_tags

        # Add computed fields from normalized metadata
        if self.litellm_mode:
            merged["mode"] = self.litellm_mode
        if self.max_tokens:
            merged["max_tokens"] = self.max_tokens
        if self.max_input_tokens:
            merged["max_input_tokens"] = self.max_input_tokens
        if self.context_window:
            merged["max_input_tokens"] = merged.get("max_input_tokens", self.context_window)
        if self.max_output_tokens:
            merged["max_output_tokens"] = self.max_output_tokens

        # Extract embedding dimensions for embedding models
        if self.model_type == "embedding" or "embedding" in self.capabilities:
            embedding_length = _extract_numeric(self.raw, "embedding_length", "embedding_size", "output_vector_size")
            if embedding_length:
                merged["output_vector_size"] = embedding_length
            # Don't set max_output_tokens for embeddings
            merged.pop("max_output_tokens", None)

        # Map capabilities to supports_* fields
        supports_fields = _map_capabilities_to_supports(self.capabilities)
        merged.update(supports_fields)

        # Add default pricing (0.0 for Ollama, normal for others)
        details = self.raw.get("details")
        ollama_family = details.get("family") if isinstance(details, dict) else None
        is_ollama_source = merged.get("litellm_provider") == "ollama" or bool(ollama_family)

        if is_ollama_source:
            # Ollama models are local/free
            merged.setdefault("litellm_provider", "ollama")
            merged.setdefault("input_cost_per_token", 0.0)
            merged.setdefault("output_cost_per_token", 0.0)
        else:
            # Use default pricing for cloud models
            pricing = _get_default_pricing(self.model_type, self.litellm_mode)
            for key, value in pricing.items():
                merged.setdefault(key, value)

        # Map supported OpenAI parameters from Ollama parameters
        supported_params = self._get_openai_compatible_params()
        if supported_params:
            merged["supported_openai_params"] = supported_params

        # Set litellm_provider based on source
        if is_ollama_source and "litellm_provider" not in merged:
            merged["litellm_provider"] = "ollama"

        # Add supports_system_messages and supports_native_streaming for chat models
        if self.model_type in ("chat", "completion") or self.litellm_mode == "chat":
            merged.setdefault("supports_system_messages", True)
            merged.setdefault("supports_native_streaming", True)

        return merged

    def _get_openai_compatible_params(self) -> list[str]:
        """Extract and map supported OpenAI-compatible parameters from model metadata."""

        params_set = set()

        # Check if raw has parameters string (Ollama format)
        params_str = self.raw.get("parameters", "")
        if isinstance(params_str, str):
            if "temperature" in params_str:
                params_set.add("temperature")
            if "top_k" in params_str:
                params_set.add("top_k")
            if "top_p" in params_str:
                params_set.add("top_p")
            if "repeat_penalty" in params_str or "frequency_penalty" in params_str:
                params_set.add("frequency_penalty")
            if "presence_penalty" in params_str:
                params_set.add("presence_penalty")
            if "stop" in params_str:
                params_set.add("stop")

        # Check modelfile for PARAMETER directives
        modelfile = self.raw.get("modelfile", "")
        if isinstance(modelfile, str):
            if "PARAMETER temperature" in modelfile:
                params_set.add("temperature")
            if "PARAMETER top_k" in modelfile:
                params_set.add("top_k")
            if "PARAMETER top_p" in modelfile:
                params_set.add("top_p")
            if "PARAMETER repeat_penalty" in modelfile:
                params_set.add("frequency_penalty")
            if "PARAMETER stop" in modelfile:
                params_set.add("stop")

        # Common parameters that most chat models support
        if self.litellm_mode == "chat" or "completion" in (self.capabilities or []):
            params_set.update(["max_tokens", "stream", "n", "seed"])

        # Add tool support if capabilities indicate it
        if "tools" in (self.capabilities or []) or "function calling" in str(
            self.capabilities or []
        ).lower():
            params_set.update(["tools", "tool_choice"])

        return sorted(params_set)


class SourceModels(BaseModel):
    """Collection of models for a source."""

    source: SourceEndpoint
    models: list[ModelMetadata] = Field(default_factory=list)
    fetched_at: datetime | None = None


class AppConfig(BaseModel):
    """Application configuration for the updater service."""

    model_config = ConfigDict(validate_assignment=True)

    litellm: LitellmDestination = Field(default_factory=LitellmDestination)
    sources: list[SourceEndpoint] = Field(default_factory=list)
    sync_interval_seconds: int = Field(
        0,
        ge=0,
        description="Sync interval in seconds (0 disables automatic synchronization)",
    )

    @model_validator(mode="after")
    def _validate_sync_interval(self) -> "AppConfig":
        """Allow disabling sync with 0 while enforcing sane intervals when enabled."""

        if 0 < self.sync_interval_seconds < 30:
            raise ValueError("Sync interval must be at least 30 seconds or 0 to disable")
        return self
