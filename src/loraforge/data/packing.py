"""Sequence multi-packing algorithm for high-efficiency LLM fine-tuning.

Implements First-Fit Decreasing (FFD) bin packing to concatenate multiple short sequences
into maximum context windows, eliminating padding waste and accelerating training throughput.
"""

from __future__ import annotations

from typing import List, Tuple
from pydantic import BaseModel, Field


class PackedSequence(BaseModel):
    """A packed sequence container combining multiple dialogue samples into one context window."""

    input_ids: List[int]
    labels: List[int]
    position_ids: List[int]
    cu_seqlens: List[int]
    sample_count: int


class PackingEfficiencyReport(BaseModel):
    """Statistical summary of sequence packing efficiency improvements."""

    original_sample_count: int
    packed_sequence_count: int
    max_seq_length: int
    total_tokens: int
    total_packed_capacity: int
    padding_tokens_avoided: int
    packing_efficiency_percent: float
    throughput_speedup_factor: float


class SequencePacker:
    """Multi-pack bin packing coordinator using First-Fit Decreasing (FFD)."""

    def __init__(self, max_seq_length: int = 2048, pad_token_id: int = 0) -> None:
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

    def pack(
        self,
        tokenized_samples: List[Tuple[List[int], List[int]]],
    ) -> Tuple[List[PackedSequence], PackingEfficiencyReport]:
        """Pack a list of (input_ids, labels) pairs into fixed-capacity sequences.

        Uses First-Fit Decreasing (FFD):
        1. Samples are sorted descending by sequence length.
        2. Each sample is placed into the first bin with sufficient remaining capacity.
        3. If no bin fits the sample, a new bin is allocated.
        """
        if not tokenized_samples:
            return [], PackingEfficiencyReport(
                original_sample_count=0,
                packed_sequence_count=0,
                max_seq_length=self.max_seq_length,
                total_tokens=0,
                total_packed_capacity=0,
                padding_tokens_avoided=0,
                packing_efficiency_percent=100.0,
                throughput_speedup_factor=1.0,
            )

        # Truncate any single sample that exceeds max_seq_length
        cleaned_samples: List[Tuple[List[int], List[int]]] = []
        total_tokens = 0
        for inp, lbl in tokenized_samples:
            if len(inp) > self.max_seq_length:
                inp = inp[: self.max_seq_length]
                lbl = lbl[: self.max_seq_length]
            cleaned_samples.append((inp, lbl))
            total_tokens += len(inp)

        # Sort descending by length for First-Fit Decreasing
        indexed_samples = sorted(
            enumerate(cleaned_samples),
            key=lambda x: len(x[1][0]),
            reverse=True,
        )

        # Bins: list of items per bin
        # Each item in bin is (input_ids, labels)
        bins: List[List[Tuple[List[int], List[int]]]] = []
        bin_remainders: List[int] = []

        for _, (inp, lbl) in indexed_samples:
            length = len(inp)
            placed = False

            # Find first bin that has enough remaining room
            for b_idx in range(len(bins)):
                if bin_remainders[b_idx] >= length:
                    bins[b_idx].append((inp, lbl))
                    bin_remainders[b_idx] -= length
                    placed = True
                    break

            if not placed:
                # Open new bin
                bins.append([(inp, lbl)])
                bin_remainders.append(self.max_seq_length - length)

        # Convert bins to PackedSequence objects
        packed_results: List[PackedSequence] = []
        for b in bins:
            all_input_ids: List[int] = []
            all_labels: List[int] = []
            all_position_ids: List[int] = []
            cu_seqlens: List[int] = [0]

            curr_offset = 0
            for inp, lbl in b:
                all_input_ids.extend(inp)
                all_labels.extend(lbl)
                # Position IDs reset to 0 for each packed sample
                all_position_ids.extend(list(range(len(inp))))
                curr_offset += len(inp)
                cu_seqlens.append(curr_offset)

            # Pad remainder if bin is not 100% full
            pad_needed = self.max_seq_length - len(all_input_ids)
            if pad_needed > 0:
                all_input_ids.extend([self.pad_token_id] * pad_needed)
                all_labels.extend([-100] * pad_needed)  # PyTorch ignore index
                all_position_ids.extend([0] * pad_needed)

            packed_results.append(
                PackedSequence(
                    input_ids=all_input_ids,
                    labels=all_labels,
                    position_ids=all_position_ids,
                    cu_seqlens=cu_seqlens,
                    sample_count=len(b),
                )
            )

        packed_count = len(packed_results)
        total_capacity = packed_count * self.max_seq_length
        unpacked_capacity = len(tokenized_samples) * self.max_seq_length
        efficiency = (total_tokens / total_capacity * 100.0) if total_capacity > 0 else 100.0
        padding_avoided = max(0, unpacked_capacity - total_capacity)
        speedup = (len(tokenized_samples) / packed_count) if packed_count > 0 else 1.0

        report = PackingEfficiencyReport(
            original_sample_count=len(tokenized_samples),
            packed_sequence_count=packed_count,
            max_seq_length=self.max_seq_length,
            total_tokens=total_tokens,
            total_packed_capacity=total_capacity,
            padding_tokens_avoided=padding_avoided,
            packing_efficiency_percent=round(efficiency, 2),
            throughput_speedup_factor=round(speedup, 2),
        )

        return packed_results, report
