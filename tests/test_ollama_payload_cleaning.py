import json
from pathlib import Path

from litellm_updater.sources import _clean_ollama_payload


def test_clean_ollama_payload_strips_large_fields():
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
