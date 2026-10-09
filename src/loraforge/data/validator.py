"""Dataset validation, token length distribution analysis, and step estimation.

Profiles token counts across training samples to inform optimal max_seq_length choices
and compute training epoch/step milestones.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional
from pydantic import BaseModel, Field

from loraforge.data.formats import TrainingSample


class TokenDistribution(BaseModel):
    """Statistical distribution of token lengths across samples."""

    total_samples: int
    total_tokens: int
    min_length: int
    mean_length: float
    median_length: float
    p90_length: float
    p95_length: float
    p99_length: float
    max_length: int
    truncated_samples: int
    truncated_percent: float


class DatasetAuditReport(BaseModel):
    """Complete validation audit report for a dataset against target sequence length."""

    is_valid: bool
    distribution: TokenDistribution
    recommended_max_seq_len: int
    estimated_total_steps: int
    warnings: List[str]


class DatasetValidator:
    """Validates dataset samples and evaluates token distribution."""

    @staticmethod
    def default_tokenizer(text: str) -> List[int]:
        """Heuristic character-to-token ratio approximation if no HuggingFace tokenizer loaded.

        Standard BPE average for English text is ~3.8 to 4.0 characters per token.
        """
        words = text.split()
        # Roughly 1.3 tokens per whitespace word
        token_count = max(1, int(len(words) * 1.3))
        return list(range(token_count))

    def audit(
        self,
        samples: List[TrainingSample],
        target_seq_len: int = 2048,
        epochs: int = 3,
        effective_batch_size: int = 8,
        use_packing: bool = True,
        tokenize_fn: Optional[Callable[[str], List[int]]] = None,
    ) -> DatasetAuditReport:
        """Run statistical audit over dataset samples."""
        if not samples:
            raise ValueError("Cannot audit empty dataset.")

        tokenizer = tokenize_fn or self.default_tokenizer
        lengths: List[int] = []
        warnings: List[str] = []

        total_tokens = 0
        truncated_count = 0

        for s in samples:
            formatted = s.to_formatted_prompt()
            toks = tokenizer(formatted)
            t_len = len(toks)
            lengths.append(t_len)
            total_tokens += t_len
            if t_len > target_seq_len:
                truncated_count += 1

        lengths.sort()
        n = len(lengths)

        def percentile(p: float) -> float:
            idx = int(math.ceil(p * n)) - 1
            idx = max(0, min(idx, n - 1))
            return float(lengths[idx])

        min_len = lengths[0]
        max_len = lengths[-1]
        mean_len = round(total_tokens / n, 2)
        med_len = percentile(0.50)
        p90 = percentile(0.90)
        p95 = percentile(0.95)
        p99 = percentile(0.99)
        trunc_pct = round((truncated_count / n) * 100.0, 2)

        dist = TokenDistribution(
            total_samples=n,
            total_tokens=total_tokens,
            min_length=min_len,
            mean_length=mean_len,
            median_length=med_len,
            p90_length=p90,
            p95_length=p95,
            p99_length=p99,
            max_length=max_len,
            truncated_samples=truncated_count,
            truncated_percent=trunc_pct,
        )

        # Warnings
        if trunc_pct > 5.0:
            warnings.append(
                f"{trunc_pct}% of samples ({truncated_count}/{n}) exceed target max_seq_length ({target_seq_len}) and will be truncated."
            )

        if min_len < 10:
            warnings.append("Dataset contains unusually short samples (< 10 tokens).")

        # Step estimation
        if use_packing:
            # Packed sequences count is approximately total_tokens / target_seq_len
            effective_sequences = max(1, math.ceil(total_tokens / target_seq_len))
        else:
            effective_sequences = n

        steps_per_epoch = max(1, math.ceil(effective_sequences / effective_batch_size))
        total_steps = steps_per_epoch * epochs

        # Recommend sequence length that covers at least p95 without excessive waste
        rec_seq = int(math.ceil(p95 / 256.0) * 256)
        rec_seq = max(512, min(rec_seq, 8192))

        return DatasetAuditReport(
            is_valid=True,
            distribution=dist,
            recommended_max_seq_len=rec_seq,
            estimated_total_steps=total_steps,
            warnings=warnings,
        )
