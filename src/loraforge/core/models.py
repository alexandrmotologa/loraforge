"""Known model architecture registry and HuggingFace config loader.

Provides pre-defined architecture specifications for popular open-weights LLMs
and utilities to introspect local HuggingFace config.json files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from loraforge.core.config import ModelArchitectureSpec

# Exact specifications for popular foundation models
PRESET_MODELS: Dict[str, ModelArchitectureSpec] = {
    # Meta Llama 3.1 8B
    "meta-llama/Llama-3.1-8B-Instruct": ModelArchitectureSpec(
        name="meta-llama/Llama-3.1-8B-Instruct",
        num_parameters=8_030_000_000,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": ModelArchitectureSpec(
        name="meta-llama/Meta-Llama-3-8B-Instruct",
        num_parameters=8_030_000_000,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    ),
    # Mistral 7B v0.3
    "mistralai/Mistral-7B-Instruct-v0.3": ModelArchitectureSpec(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        num_parameters=7_250_000_000,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32768,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    ),
    # Qwen 2.5 7B
    "Qwen/Qwen2.5-7B-Instruct": ModelArchitectureSpec(
        name="Qwen/Qwen2.5-7B-Instruct",
        num_parameters=7_610_000_000,
        num_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=152064,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    ),
    # Qwen 2.5 14B
    "Qwen/Qwen2.5-14B-Instruct": ModelArchitectureSpec(
        name="Qwen/Qwen2.5-14B-Instruct",
        num_parameters=14_770_000_000,
        num_layers=48,
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_key_value_heads=8,
        vocab_size=152064,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    ),
    # Microsoft Phi-3.5-mini
    "microsoft/Phi-3.5-mini-instruct": ModelArchitectureSpec(
        name="microsoft/Phi-3.5-mini-instruct",
        num_parameters=3_820_000_000,
        num_layers=32,
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_key_value_heads=32,
        vocab_size=32064,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
    ),
    # Google Gemma 2 9B
    "google/gemma-2-9b-it": ModelArchitectureSpec(
        name="google/gemma-2-9b-it",
        num_parameters=9_240_000_000,
        num_layers=42,
        hidden_size=3584,
        intermediate_size=14336,
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=256000,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
    ),
    # DeepSeek R1 Distill Qwen 7B
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelArchitectureSpec(
        name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        num_parameters=7_610_000_000,
        num_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=152064,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    ),
    # DeepSeek R1 Distill Llama 8B
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ModelArchitectureSpec(
        name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        num_parameters=8_030_000_000,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
    ),
    # Meta Llama 3.1 70B
    "meta-llama/Llama-3.1-70B-Instruct": ModelArchitectureSpec(
        name="meta-llama/Llama-3.1-70B-Instruct",
        num_parameters=70_600_000_000,
        num_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
    ),
}


def get_model_spec(model_name_or_path: str) -> Optional[ModelArchitectureSpec]:
    """Retrieve architecture spec from preset registry or local directory."""
    # 1. Direct registry hit
    if model_name_or_path in PRESET_MODELS:
        return PRESET_MODELS[model_name_or_path]

    # 2. Case-insensitive or partial match in registry
    normalized = model_name_or_path.lower()
    for key, spec in PRESET_MODELS.items():
        if key.lower() == normalized or key.split("/")[-1].lower() == normalized:
            return spec

    # 3. Check for local config.json
    local_path = Path(model_name_or_path)
    config_file = local_path / "config.json" if local_path.is_dir() else local_path
    if config_file.is_file():
        return parse_huggingface_config(config_file)

    return None


def parse_huggingface_config(config_path: Path | str) -> ModelArchitectureSpec:
    """Parse a HuggingFace transformers config.json into ModelArchitectureSpec."""
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Standard transformers config fields
    num_layers = cfg.get("num_hidden_layers", cfg.get("n_layer", 32))
    hidden_size = cfg.get("hidden_size", cfg.get("n_embd", 4096))
    intermediate_size = cfg.get(
        "intermediate_size",
        cfg.get("n_inner", int(hidden_size * 3.5)),
    )
    num_heads = cfg.get("num_attention_heads", cfg.get("n_head", 32))
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    vocab_size = cfg.get("vocab_size", 32000)
    max_pos = cfg.get("max_position_embeddings", 4096)
    tie_embeddings = cfg.get("tie_word_embeddings", False)

    # Estimate parameter count if not provided
    # Embeddings: vocab_size * hidden_size
    # Transformer blocks: L * (attn + mlp + norms)
    # attn: q (h*h) + k (h*kv_dim) + v (h*kv_dim) + o (h*h)
    kv_dim = (hidden_size // num_heads) * num_kv_heads
    attn_params = hidden_size * (hidden_size + 2 * kv_dim + hidden_size)
    mlp_params = 3 * hidden_size * intermediate_size  # SwiGLU: gate, up, down
    layer_params = attn_params + mlp_params + (2 * hidden_size)
    total_params = (vocab_size * hidden_size) + (num_layers * layer_params)
    if not tie_embeddings:
        total_params += vocab_size * hidden_size

    return ModelArchitectureSpec(
        name=cfg.get("_name_or_path", path.stem),
        num_parameters=total_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        tie_word_embeddings=tie_embeddings,
    )
