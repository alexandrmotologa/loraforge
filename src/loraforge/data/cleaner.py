"""Dataset sanitization, deduplication, and quality filtering engine.

Cleans messy, web-scraped, or LLM-generated training datasets:
- Strips invisible Unicode artifacts and redundant whitespace.
- Drops exact duplicate samples via SHA-256 content hashing.
- Removes degenerate single-turn or empty conversations.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from loraforge.core.config import DatasetFormat
from loraforge.data.formats import DatasetParser, DialogueTurn, TrainingSample


class DataCleanReport(BaseModel):
    """Statistical summary of dataset sanitization results."""

    total_input_samples: int
    retained_samples: int
    duplicate_samples_dropped: int
    degenerate_samples_dropped: int
    whitespace_trimmed_turns: int
    retention_rate_pct: float
    output_path: str


class DatasetCleaner:
    """Sanitizes conversational datasets and eliminates low-quality samples."""

    # Matches invisible control characters except newline and tab
    CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\ufeff]")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Strip invisible control characters, normalize newlines, and trim whitespace."""
        cleaned = cls.CONTROL_CHAR_REGEX.sub("", text)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        return cleaned.strip()

    @classmethod
    def hash_sample(cls, sample: TrainingSample) -> str:
        """Compute deterministic SHA-256 fingerprint for deduplication."""
        hasher = hashlib.sha256()
        for turn in sample.messages:
            hasher.update(turn.role.lower().encode("utf-8"))
            hasher.update(b":")
            hasher.update(turn.content.strip().encode("utf-8"))
            hasher.update(b"|")
        return hasher.hexdigest()

    @classmethod
    def clean(
        cls,
        input_path: Path | str,
        output_path: Path | str,
        fmt: DatasetFormat = DatasetFormat.SHAREGPT,
        min_turns: int = 2,
        require_assistant_turn: bool = True,
    ) -> DataCleanReport:
        """Process dataset file, apply sanitization rules, and write clean JSONL output."""
        in_file = Path(input_path)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        samples = DatasetParser.load_jsonl(in_file, fmt)
        total_input = len(samples)

        seen_hashes: Set[str] = set()
        cleaned_samples: List[TrainingSample] = []

        duplicate_count = 0
        degenerate_count = 0
        trimmed_turns_count = 0

        for s in samples:
            # 1. Clean turn texts and drop empty turns
            valid_turns: List[DialogueTurn] = []
            for m in s.messages:
                original = m.content
                cleaned = cls.clean_text(original)
                if cleaned != original:
                    trimmed_turns_count += 1

                if cleaned:
                    valid_turns.append(DialogueTurn(role=m.role.lower().strip(), content=cleaned))

            # 2. Check minimum turns
            if len(valid_turns) < min_turns:
                degenerate_count += 1
                continue

            # 3. Check assistant response presence
            if require_assistant_turn:
                has_assistant = any(t.role in ("assistant", "gpt", "model") for t in valid_turns)
                if not has_assistant:
                    degenerate_count += 1
                    continue

            candidate = TrainingSample(messages=valid_turns)

            # 4. Deduplication
            fp = cls.hash_sample(candidate)
            if fp in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(fp)
            cleaned_samples.append(candidate)

        # Write output in standard JSONL (ShareGPT / OpenAI format)
        with open(out_file, "w", encoding="utf-8") as f:
            for s in cleaned_samples:
                record = {
                    "messages": [t.model_dump() for t in s.messages]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        retained = len(cleaned_samples)
        retention_pct = round((retained / total_input * 100.0), 2) if total_input > 0 else 100.0

        return DataCleanReport(
            total_input_samples=total_input,
            retained_samples=retained,
            duplicate_samples_dropped=duplicate_count,
            degenerate_samples_dropped=degenerate_count,
            whitespace_trimmed_turns=trimmed_turns_count,
            retention_rate_pct=retention_pct,
            output_path=str(out_file.resolve()),
        )
