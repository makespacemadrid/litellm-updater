"""Default OpenAI-compatible models configuration.

This module defines the standard set of OpenAI model aliases that map to
Ollama and LocalAI models, providing drop-in compatibility for applications
using the OpenAI API format.
"""

# Default provider base URLs
DEFAULT_OLLAMA_BASE = "https://olm.mksmad.org"
DEFAULT_LOCALAI_BASE = "http://10.0.8.100:8081"

# Complete list of OpenAI-compatible model definitions
DEFAULT_COMPAT_MODELS = [
    # ========== CHAT MODELS ==========
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "ollama/qwen3:4b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "fast"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "fast", "family:qwen3", "size:4B"],
        },
    },
    {
        "model_name": "gpt-3.5-turbo-16k",
        "litellm_params": {
            "model": "ollama/qwen3:8b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "fast"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "fast", "family:qwen3", "size:8.2B"],
        },
    },
    {
        "model_name": "gpt-3.5-turbo-instruct",
        "litellm_params": {
            "model": "ollama/qwen2.5-coder:7b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "instruct", "code"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_function_calling": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "instruct", "code", "family:qwen2.5", "size:7B"],
        },
    },
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "ollama/qwen3:8b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "balanced"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "balanced", "family:qwen3", "size:8.2B"],
        },
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "ollama/qwen3:32b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "premium"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "premium", "family:qwen3", "size:32.8B"],
        },
    },
    {
        "model_name": "gpt-4-32k",
        "litellm_params": {
            "model": "ollama/llama3.1:8b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "large-context"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "large-context", "family:llama", "size:8B"],
        },
    },
    {
        "model_name": "gpt-4-turbo",
        "litellm_params": {
            "model": "ollama/llama3.3:70b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "best-quality"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 8192,
            "max_input_tokens": 8192,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "best-quality", "family:llama", "size:70B"],
        },
    },
    {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "ollama/qwen3:32b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "chat", "premium"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "chat", "premium", "family:qwen3", "size:32.8B"],
        },
    },
    # ========== VISION MODELS ==========
    {
        "model_name": "gpt-4-vision-preview",
        "litellm_params": {
            "model": "ollama/llama3.2-vision:11b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "vision"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_vision": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "vision", "stable", "family:mllama", "size:10.7B"],
        },
    },
    {
        "model_name": "gpt-4-turbo-vision",
        "litellm_params": {
            "model": "ollama/qwen3-vl:8b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "vision", "reasoning"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "vision", "reasoning", "family:qwen3vl", "size:8B"],
        },
    },
    {
        "model_name": "gpt-4o-vision",
        "litellm_params": {
            "model": "ollama/qwen3-vl:32b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "vision", "premium"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "vision", "premium", "family:qwen3vl", "size:32B"],
        },
    },
    # ========== EMBEDDING MODELS ==========
    {
        "model_name": "text-embedding-3-small",
        "litellm_params": {
            "model": "ollama/nomic-embed-text:latest",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "embedding", "fast"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "embedding",
            "max_input_tokens": 2048,
            "output_vector_size": 768,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "tags": ["compat", "openai-alias", "embedding", "fast", "family:nomic-bert", "size:137M"],
        },
    },
    {
        "model_name": "text-embedding-ada-002",
        "litellm_params": {
            "model": "ollama/nomic-embed-text:latest",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "embedding", "legacy"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "embedding",
            "max_input_tokens": 2048,
            "output_vector_size": 768,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "tags": ["compat", "openai-alias", "embedding", "legacy", "family:nomic-bert", "size:137M"],
        },
    },
    {
        "model_name": "text-embedding-3-large",
        "litellm_params": {
            "model": "ollama/qwen3-embedding:8b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "embedding", "quality"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "embedding",
            "max_input_tokens": 32768,
            "output_vector_size": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "tags": ["compat", "openai-alias", "embedding", "quality", "family:qwen3", "size:7.6B"],
        },
    },
    # ========== REASONING MODELS ==========
    {
        "model_name": "o1-mini",
        "litellm_params": {
            "model": "ollama/deepseek-r1:7b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "reasoning", "fast"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "reasoning", "fast", "family:deepseek", "size:7.6B"],
        },
    },
    {
        "model_name": "o1-preview",
        "litellm_params": {
            "model": "ollama/deepseek-r1:14b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "reasoning", "balanced"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "reasoning", "balanced", "family:deepseek", "size:14B"],
        },
    },
    {
        "model_name": "o1",
        "litellm_params": {
            "model": "ollama/deepseek-r1:32b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "reasoning", "premium"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_reasoning": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "reasoning", "premium", "family:deepseek", "size:32.8B"],
        },
    },
    # ========== CODE MODELS ==========
    {
        "model_name": "code-cushman-001",
        "litellm_params": {
            "model": "ollama/qwen2.5-coder:1.5b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "code", "lightweight"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_function_calling": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "code", "lightweight", "family:qwen2.5", "size:1.5B"],
        },
    },
    {
        "model_name": "code-davinci-002",
        "litellm_params": {
            "model": "ollama/qwen2.5-coder:14b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "code", "balanced"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_function_calling": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "code", "balanced", "family:qwen2.5", "size:14B"],
        },
    },
    {
        "model_name": "gpt-4.1",
        "litellm_params": {
            "model": "ollama/qwen3-coder:30b",
            "api_base": DEFAULT_OLLAMA_BASE,
            "tags": ["compat", "openai-alias", "code", "premium"],
        },
        "model_info": {
            "litellm_provider": "ollama",
            "mode": "chat",
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_system_messages": True,
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_native_streaming": True,
            "tags": ["compat", "openai-alias", "code", "premium", "family:qwen3", "size:30B"],
        },
    },
    # ========== IMAGE GENERATION MODELS ==========
    {
        "model_name": "dall-e-2",
        "litellm_params": {
            "model": "openai/stablediffusion",
            "api_base": DEFAULT_LOCALAI_BASE,
            "tags": ["compat", "openai-alias", "image-generation", "standard"],
        },
        "model_info": {
            "litellm_provider": "openai",
            "mode": "image_generation",
            "max_images": 1,
            "default_size": "512x512",
            "supported_sizes": ["256x256", "512x512", "1024x1024"],
            "output_cost_per_image": 0.0,
            "tags": ["compat", "openai-alias", "image-gen", "stable-diffusion"],
        },
    },
    {
        "model_name": "dall-e-3",
        "litellm_params": {
            "model": "openai/dreamshaper",
            "api_base": DEFAULT_LOCALAI_BASE,
            "tags": ["compat", "openai-alias", "image-generation", "premium"],
        },
        "model_info": {
            "litellm_provider": "openai",
            "mode": "image_generation",
            "max_images": 1,
            "default_size": "512x512",
            "supported_sizes": ["512x512", "1024x1024"],
            "output_cost_per_image": 0.0,
            "tags": ["compat", "openai-alias", "image-gen", "dreamshaper", "premium"],
        },
    },
]


def get_compat_models_by_category():
    """Return models grouped by category."""
    categories = {
        "chat": [],
        "vision": [],
        "embedding": [],
        "reasoning": [],
        "code": [],
        "image_generation": [],
    }

    for model in DEFAULT_COMPAT_MODELS:
        tags = model["litellm_params"]["tags"]
        if "image-generation" in tags:
            categories["image_generation"].append(model)
        elif "vision" in tags:
            categories["vision"].append(model)
        elif "embedding" in tags:
            categories["embedding"].append(model)
        elif "reasoning" in tags:
            categories["reasoning"].append(model)
        elif "code" in tags:
            categories["code"].append(model)
        else:
            categories["chat"].append(model)

    return categories


def get_model_count_summary():
    """Return summary of model counts by category."""
    categories = get_compat_models_by_category()
    return {cat: len(models) for cat, models in categories.items()}
