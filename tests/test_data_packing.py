"""Unit tests for dataset parsing, validation, and multi-pack algorithms."""

import pytest
from loraforge.core.config import DatasetFormat
from loraforge.data.formats import DatasetParser, DialogueTurn, TrainingSample
from loraforge.data.packing import SequencePacker
from loraforge.data.validator import DatasetValidator


def test_parse_alpaca_record():
    raw = {
        "instruction": "Summarize the text.",
        "input": "Antigravity accelerates AI code development.",
        "output": "Antigravity helps developers build software.",
    }
    sample = DatasetParser.parse_record(raw, DatasetFormat.ALPACA)
    assert len(sample.messages) == 2
    assert sample.messages[0].role == "user"
    assert "Input:" in sample.messages[0].content
    assert sample.messages[1].role == "assistant"


def test_parse_sharegpt_record():
    raw = {
        "conversations": [
            {"from": "human", "value": "What is QLoRA?"},
            {"from": "gpt", "value": "QLoRA is Quantized Low-Rank Adaptation."},
        ]
    }
    sample = DatasetParser.parse_record(raw, DatasetFormat.SHAREGPT)
    assert len(sample.messages) == 2
    assert sample.messages[0].role == "user"
    assert sample.messages[1].role == "assistant"


def test_parse_chatml_record():
    raw = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    sample = DatasetParser.parse_record(raw, DatasetFormat.CHATML)
    assert len(sample.messages) == 3
    assert sample.messages[0].role == "system"


def test_sequence_packer_ffd():
    # 3 short samples that fit within max_seq_length = 50
    # Lengths: 15, 20, 10
    sample1 = ([1] * 15, [2] * 15)
    sample2 = ([3] * 20, [4] * 20)
    sample3 = ([5] * 10, [6] * 10)

    packer = SequencePacker(max_seq_length=50)
    packed, report = packer.pack([sample1, sample2, sample3])

    # All three samples should pack into a single bin of 50 tokens (15 + 20 + 10 = 45 + 5 pad)
    assert len(packed) == 1
    assert packed[0].sample_count == 3
    assert len(packed[0].input_ids) == 50
    assert packed[0].cu_seqlens == [0, 20, 35, 45]
    # Check position IDs reset
    assert packed[0].position_ids[0] == 0  # start of sample2
    assert packed[0].position_ids[20] == 0  # start of sample1
    assert packed[0].position_ids[35] == 0  # start of sample3

    assert report.padding_tokens_avoided > 0
    assert report.packing_efficiency_percent == 90.0  # 45 real / 50 total


def test_dataset_validator_audit():
    samples = [
        TrainingSample(messages=[
            DialogueTurn(role="user", content="Short prompt."),
            DialogueTurn(role="assistant", content="Short answer."),
        ]),
        TrainingSample(messages=[
            DialogueTurn(role="user", content="Explain quantum computing in detail." * 20),
            DialogueTurn(role="assistant", content="Quantum computing uses qubits and superposition." * 20),
        ]),
    ]

    validator = DatasetValidator()
    report = validator.audit(samples, target_seq_len=1024, epochs=3, effective_batch_size=2)

    assert report.is_valid is True
    assert report.distribution.total_samples == 2
    assert report.distribution.min_length < report.distribution.max_length
    assert report.estimated_total_steps >= 1
