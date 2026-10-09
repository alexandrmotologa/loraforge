"""Unit and integration tests for dataset cleaner and chat templates engine."""

import json
from pathlib import Path
import pytest

from loraforge.core.config import DatasetFormat
from loraforge.data.cleaner import DatasetCleaner, DataCleanReport
from loraforge.data.formats import DatasetParser, DialogueTurn, TrainingSample
from loraforge.data.templates import (
    TEMPLATES,
    ChatTemplate,
    format_chatml,
    format_deepseek,
    format_gemma2,
    format_llama3,
    format_mistral,
    format_phi3,
    get_template,
)


def test_clean_text_normalization():
    raw = "\ufeffHello\x00 world!\u200b\r\nThis is a test.\r   "
    cleaned = DatasetCleaner.clean_text(raw)
    assert cleaned == "Hello world!\nThis is a test."
    assert "\x00" not in cleaned
    assert "\ufeff" not in cleaned
    assert "\u200b" not in cleaned
    assert "\r" not in cleaned


def test_hash_sample_deterministic():
    s1 = TrainingSample(messages=[
        DialogueTurn(role="user", content="What is QLoRA?"),
        DialogueTurn(role="assistant", content="Quantized Low-Rank Adaptation."),
    ])
    s2 = TrainingSample(messages=[
        DialogueTurn(role="user", content="  What is QLoRA?  "),
        DialogueTurn(role="assistant", content="Quantized Low-Rank Adaptation."),
    ])
    s3 = TrainingSample(messages=[
        DialogueTurn(role="user", content="What is LoRA?"),
        DialogueTurn(role="assistant", content="Low-Rank Adaptation."),
    ])

    h1 = DatasetCleaner.hash_sample(s1)
    h2 = DatasetCleaner.hash_sample(s2)
    h3 = DatasetCleaner.hash_sample(s3)

    assert h1 == h2
    assert h1 != h3


def test_dataset_cleaner_pipeline(tmp_path: Path):
    in_file = tmp_path / "dirty.jsonl"
    out_file = tmp_path / "cleaned" / "clean.jsonl"

    raw_data = [
        # 1. Valid sample
        {
            "conversations": [
                {"from": "human", "value": "Calculate 2+2."},
                {"from": "gpt", "value": "2+2 equals 4."},
            ]
        },
        # 2. Sample with control characters and whitespace
        {
            "conversations": [
                {"from": "human", "value": "  \ufeffDefine inertia.\u200b  "},
                {"from": "gpt", "value": "Inertia is resistance to change in motion.\r\n"},
            ]
        },
        # 3. Exact duplicate of sample 1
        {
            "conversations": [
                {"from": "human", "value": "Calculate 2+2."},
                {"from": "gpt", "value": "2+2 equals 4."},
            ]
        },
        # 4. Degenerate sample with only 1 turn (min_turns=2)
        {
            "conversations": [
                {"from": "human", "value": "Hello alone"},
            ]
        },
        # 5. Degenerate sample without assistant turn
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "human", "value": "Is anyone there?"},
            ]
        },
    ]

    with open(in_file, "w", encoding="utf-8") as f:
        for item in raw_data:
            f.write(json.dumps(item) + "\n")

    report: DataCleanReport = DatasetCleaner.clean(
        in_file,
        out_file,
        fmt=DatasetFormat.SHAREGPT,
        min_turns=2,
        require_assistant_turn=True,
    )

    assert report.total_input_samples == 5
    assert report.retained_samples == 2
    assert report.duplicate_samples_dropped == 1
    assert report.degenerate_samples_dropped == 2
    assert report.whitespace_trimmed_turns >= 1
    assert report.retention_rate_pct == 40.0
    assert out_file.is_file()

    # Verify that clean file can be reloaded
    reloaded = DatasetParser.load_jsonl(out_file, DatasetFormat.OPENAI_MESSAGES)
    assert len(reloaded) == 2
    assert reloaded[0].messages[0].content == "Calculate 2+2."
    assert reloaded[1].messages[0].content == "Define inertia."


def test_chat_template_llama3():
    sample = TrainingSample(messages=[
        DialogueTurn(role="system", content="You are a helpful assistant."),
        DialogueTurn(role="user", content="Explain quantum computing."),
        DialogueTurn(role="assistant", content="Quantum computing uses qubits."),
    ])
    formatted = format_llama3(sample)
    assert formatted.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>" in formatted
    assert "<|start_header_id|>user<|end_header_id|>\n\nExplain quantum computing.<|eot_id|>" in formatted
    assert "<|start_header_id|>assistant<|end_header_id|>\n\nQuantum computing uses qubits.<|eot_id|>" in formatted


def test_chat_template_chatml():
    sample = TrainingSample(messages=[
        DialogueTurn(role="user", content="Hello"),
        DialogueTurn(role="assistant", content="Hi there!"),
    ])
    formatted = format_chatml(sample)
    assert "<|im_start|>user\nHello<|im_end|>" in formatted
    assert "<|im_start|>assistant\nHi there!<|im_end|>" in formatted


def test_chat_template_mistral():
    sample = TrainingSample(messages=[
        DialogueTurn(role="system", content="Act as a philosopher."),
        DialogueTurn(role="user", content="What is truth?"),
        DialogueTurn(role="assistant", content="Truth is reality."),
    ])
    formatted = format_mistral(sample)
    assert formatted.startswith("<s>")
    assert "[INST] Act as a philosopher.\n\nWhat is truth? [/INST]" in formatted
    assert " Truth is reality.</s>" in formatted


def test_chat_template_deepseek():
    sample = TrainingSample(messages=[
        DialogueTurn(role="system", content="Solve problems step by step."),
        DialogueTurn(role="user", content="What is 15 * 6?"),
        DialogueTurn(role="assistant", content="15 * 6 = 90."),
    ])
    formatted = format_deepseek(sample)
    assert formatted.startswith("<｜begin of sentence｜>")
    assert "Solve problems step by step.\n\n" in formatted
    assert "<｜User｜>What is 15 * 6?" in formatted
    assert "<｜Assistant｜>15 * 6 = 90.<｜end of sentence｜>" in formatted


def test_chat_template_gemma2():
    sample = TrainingSample(messages=[
        DialogueTurn(role="user", content="What is Python?"),
        DialogueTurn(role="assistant", content="Python is a language."),
    ])
    formatted = format_gemma2(sample)
    assert formatted.startswith("<bos>")
    assert "<start_of_turn>user\nWhat is Python?<end_of_turn>" in formatted
    assert "<start_of_turn>model\nPython is a language.<end_of_turn>" in formatted


def test_chat_template_phi3():
    sample = TrainingSample(messages=[
        DialogueTurn(role="system", content="System instruction."),
        DialogueTurn(role="user", content="User prompt."),
        DialogueTurn(role="assistant", content="Model response."),
    ])
    formatted = format_phi3(sample)
    assert formatted.startswith("<s>")
    assert "<|system|>\nSystem instruction.<|end|>" in formatted
    assert "<|user|>\nUser prompt.<|end|>" in formatted
    assert "<|assistant|>\nModel response.<|end|>" in formatted


def test_get_template_lookup_and_fallback():
    assert get_template("llama3").name == "llama3"
    assert get_template("llama-3").name == "llama3"
    assert get_template("chatml").name == "chatml"
    assert get_template("mistral").name == "mistral"
    assert get_template("deepseek").name == "deepseek"
    assert get_template("gemma2").name == "gemma2"
    assert get_template("phi3").name == "phi3"
    # Fallback to chatml for unknown template
    assert get_template("unknown-special-template").name == "chatml"
