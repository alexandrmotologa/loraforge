"""Chat template registry and conversation formatter.

Formats multi-turn conversations into native prompt representations for major model families
(Llama 3, ChatML, Mistral, DeepSeek-R1, Gemma 2, Phi 3).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional
from loraforge.data.formats import TrainingSample


class ChatTemplate:
    """Formatter specification for conversational special tokens."""

    def __init__(self, name: str, format_fn: Callable[[TrainingSample], str]) -> None:
        self.name = name
        self.format_fn = format_fn

    def format(self, sample: TrainingSample) -> str:
        return self.format_fn(sample)


def format_llama3(sample: TrainingSample) -> str:
    """Meta Llama 3 / 3.1 Instruct chat template."""
    chunks = ["<|begin_of_text|>"]
    for turn in sample.messages:
        role = turn.role.lower()
        chunks.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{turn.content}<|eot_id|>")
    return "".join(chunks)


def format_chatml(sample: TrainingSample) -> str:
    """Standard ChatML template (Qwen, Hermes, Yi)."""
    chunks = []
    for turn in sample.messages:
        role = turn.role.lower()
        chunks.append(f"<|im_start|>{role}\n{turn.content}<|im_end|>")
    return "\n".join(chunks)


def format_mistral(sample: TrainingSample) -> str:
    """Mistral Instruct template."""
    chunks = ["<s>"]
    system_text = ""
    for turn in sample.messages:
        role = turn.role.lower()
        if role == "system":
            system_text = turn.content.strip() + "\n\n"
        elif role == "user":
            user_content = f"{system_text}{turn.content}" if system_text else turn.content
            chunks.append(f"[INST] {user_content} [/INST]")
            system_text = ""
        elif role in ("assistant", "model"):
            chunks.append(f" {turn.content}</s>")
    return "".join(chunks)


def format_deepseek(sample: TrainingSample) -> str:
    """DeepSeek-V2 / R1 format template."""
    chunks = ["<｜begin of sentence｜>"]
    for turn in sample.messages:
        role = turn.role.lower()
        if role == "system":
            chunks.append(f"{turn.content}\n\n")
        elif role == "user":
            chunks.append(f"<｜User｜>{turn.content}")
        elif role in ("assistant", "model"):
            chunks.append(f"<｜Assistant｜>{turn.content}<｜end of sentence｜>")
    return "".join(chunks)


def format_gemma2(sample: TrainingSample) -> str:
    """Google Gemma 2 chat template."""
    chunks = ["<bos>"]
    for turn in sample.messages:
        role = turn.role.lower()
        role_label = "model" if role in ("assistant", "gpt") else "user"
        chunks.append(f"<start_of_turn>{role_label}\n{turn.content}<end_of_turn>\n")
    return "".join(chunks).strip()


def format_phi3(sample: TrainingSample) -> str:
    """Microsoft Phi-3 / 3.5 chat template."""
    chunks = ["<s>"]
    for turn in sample.messages:
        role = turn.role.lower()
        if role == "system":
            chunks.append(f"<|system|>\n{turn.content}<|end|>\n")
        elif role == "user":
            chunks.append(f"<|user|>\n{turn.content}<|end|>\n")
        elif role in ("assistant", "model"):
            chunks.append(f"<|assistant|>\n{turn.content}<|end|>\n")
    return "".join(chunks).strip()


TEMPLATES: Dict[str, ChatTemplate] = {
    "llama3": ChatTemplate("llama3", format_llama3),
    "chatml": ChatTemplate("chatml", format_chatml),
    "mistral": ChatTemplate("mistral", format_mistral),
    "deepseek": ChatTemplate("deepseek", format_deepseek),
    "gemma2": ChatTemplate("gemma2", format_gemma2),
    "phi3": ChatTemplate("phi3", format_phi3),
}


def get_template(name: str) -> Optional[ChatTemplate]:
    """Retrieve chat template by name or alias."""
    norm = name.lower().replace("-", "").replace("_", "")
    for k, t in TEMPLATES.items():
        if k == norm or norm in k:
            return t
    return TEMPLATES.get("chatml")
