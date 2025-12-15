"""Model categorization helpers for statistics and visualization."""
import json


def categorize_model(capabilities_json: str | None, system_tags_json: str | None) -> str:
    """
    Categorize a model based on its capabilities and tags.

    Returns one of: chat, vision, code, reasoning, embedding, audio, image, unknown

    Priority order (first match wins):
    1. Vision (vision capability)
    2. Embedding (embedding capability)
    3. Audio (audio capability)
    4. Reasoning (thinking capability or o1/o3 in tags)
    5. Code (code-related tags)
    6. Image (image/dall-e tags)
    7. Chat (default for completion/chat/function-calling)
    """
    # Parse JSON fields
    capabilities = []
    if capabilities_json:
        try:
            capabilities = json.loads(capabilities_json)
        except (json.JSONDecodeError, TypeError):
            capabilities = []

    tags = []
    if system_tags_json:
        try:
            tags = json.loads(system_tags_json)
        except (json.JSONDecodeError, TypeError):
            tags = []

    # Normalize to lowercase for comparison
    caps_lower = [c.lower() if isinstance(c, str) else str(c).lower() for c in capabilities]
    tags_lower = [t.lower() if isinstance(t, str) else str(t).lower() for t in tags]

    # Priority-based categorization
    if "vision" in caps_lower:
        return "vision"

    if "embedding" in caps_lower:
        return "embedding"

    if "audio" in caps_lower:
        return "audio"

    # Reasoning: thinking capability or o1/o3 in model name
    if "thinking" in caps_lower:
        return "reasoning"
    for tag in tags_lower:
        if any(kw in tag for kw in ["model:o1", "model:o3", "reasoning"]):
            return "reasoning"

    # Code: coder/code in model name or tags
    for tag in tags_lower:
        if any(kw in tag for kw in ["coder", "code-", "codellama", "starcoder", "deepseek-coder"]):
            return "code"

    # Image: dall-e or image generation
    for tag in tags_lower:
        if any(kw in tag for kw in ["dall-e", "dalle", "stable-diffusion", "flux"]):
            return "image"

    # Default to chat for completion/chat/function-calling
    if any(cap in caps_lower for cap in ["chat", "completion", "function calling", "tools"]):
        return "chat"

    return "unknown"


def get_category_stats(models_data: list[dict]) -> dict[str, int]:
    """
    Calculate category statistics from a list of model dicts.

    Args:
        models_data: List of dicts with 'capabilities' and 'system_tags' keys

    Returns:
        Dict mapping category name to count
    """
    stats = {
        "chat": 0,
        "vision": 0,
        "code": 0,
        "reasoning": 0,
        "embedding": 0,
        "audio": 0,
        "image": 0,
        "unknown": 0,
    }

    for model in models_data:
        category = categorize_model(
            model.get("capabilities"),
            model.get("system_tags")
        )
        stats[category] += 1

    return stats
