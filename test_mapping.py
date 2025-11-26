#!/usr/bin/env python3
"""Test script to verify Ollama to LiteLLM field mapping with sample data."""

import json
from litellm_updater.models import ModelMetadata


def test_ollama_to_litellm_mapping():
    """Test mapping of Ollama model data to LiteLLM-compatible format."""

    # Load sample Ollama data
    with open("datasamples/ollama_modelinfo_sample.json") as f:
        ollama_samples = json.load(f)

    print("=" * 80)
    print("TESTING OLLAMA → LITELLM FIELD MAPPING")
    print("=" * 80)
    print()

    for idx, raw_model in enumerate(ollama_samples, 1):
        # Extract model ID from capabilities or use index
        model_id = raw_model.get("capabilities", [f"test-model-{idx}"])[0] if raw_model.get("capabilities") else f"test-model-{idx}"
        if "modified_at" in raw_model:
            # This looks like a real model entry with details
            details = raw_model.get("details", {})
            families = details.get("families", [])
            param_size = details.get("parameter_size", "")
            model_id = f"{families[0] if families else 'unknown'}:{param_size}".lower()

        print(f"Model #{idx}: {model_id}")
        print("-" * 80)

        # Create ModelMetadata from raw data
        metadata = ModelMetadata.from_raw(model_id, raw_model)

        # Get LiteLLM-mappable fields
        litellm_fields = metadata.litellm_mappable

        # Display key information
        print(f"  Model Type: {metadata.model_type}")
        print(f"  Mode: {metadata.mode}")
        print(f"  Capabilities: {metadata.capabilities}")
        print(f"  Context Window: {metadata.context_window}")
        print(f"  Max Output Tokens: {metadata.max_output_tokens}")
        print()

        # Display pricing
        print("  Pricing (default based on OpenAI):")
        if "input_cost_per_token" in litellm_fields:
            print(f"    Input: ${litellm_fields['input_cost_per_token'] * 1000:.4f}/1K tokens")
        if "output_cost_per_token" in litellm_fields:
            print(f"    Output: ${litellm_fields['output_cost_per_token'] * 1000:.4f}/1K tokens")
        if "input_cost_per_second" in litellm_fields:
            print(f"    Audio: ${litellm_fields['input_cost_per_second'] * 60:.4f}/minute")
        if "output_cost_per_image" in litellm_fields:
            print(f"    Image: ${litellm_fields['output_cost_per_image']:.2f}/image")
        print()

        # Display supports fields
        supports_fields = {k: v for k, v in litellm_fields.items() if k.startswith("supports_")}
        if supports_fields:
            print("  Supports:")
            for key, value in sorted(supports_fields.items()):
                field_name = key.replace("supports_", "").replace("_", " ").title()
                print(f"    {field_name}: {value}")
            print()

        # Display supported OpenAI params
        if "supported_openai_params" in litellm_fields:
            params = litellm_fields["supported_openai_params"]
            print(f"  Supported OpenAI Parameters: {', '.join(params)}")
            print()

        # Display provider
        print(f"  LiteLLM Provider: {litellm_fields.get('litellm_provider', 'N/A')}")
        print()

        # Show full mappable fields (abbreviated)
        print("  All LiteLLM-mappable fields:")
        for key in sorted(litellm_fields.keys()):
            if key not in ["supported_openai_params"] and not key.startswith("supports_"):
                value = litellm_fields[key]
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
                elif isinstance(value, str) and len(value) < 50:
                    print(f"    {key}: {value}")
        print()
        print()


if __name__ == "__main__":
    test_ollama_to_litellm_mapping()
