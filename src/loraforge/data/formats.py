"""Dataset parsing, schema normalization, and prompt templating.

Converts Alpaca, ShareGPT, OpenAI Messages, ChatML, and raw text formats
into a unified conversational turn representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from loraforge.core.config import DatasetFormat
from loraforge.core.errors import DatasetValidationError


class DialogueTurn(BaseModel):
    """A single dialogue exchange turn."""

    role: str = Field(description="Role identifier: 'system', 'user', or 'assistant'.")
    content: str = Field(description="Textual payload of the message.")


class TrainingSample(BaseModel):
    """Normalized training sample consisting of an ordered list of dialogue turns."""

    id: Optional[str] = None
    messages: List[DialogueTurn]

    def to_formatted_prompt(self, template: str = "chatml") -> str:
        """Format the sample into a formatted string ready for tokenization."""
        if template == "chatml":
            chunks = []
            for m in self.messages:
                chunks.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
            return "\n".join(chunks)

        if template == "llama3":
            chunks = ["<|begin_of_text|>"]
            for m in self.messages:
                chunks.append(
                    f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>"
                )
            return "".join(chunks)

        # Fallback raw transcript
        chunks = []
        for m in self.messages:
            chunks.append(f"### {m.role.capitalize()}:\n{m.content}")
        return "\n\n".join(chunks)


class DatasetParser:
    """Parses heterogeneous JSON/JSONL datasets into normalized TrainingSample objects."""

    @staticmethod
    def parse_record(record: Dict[str, Any], fmt: DatasetFormat) -> TrainingSample:
        """Parse a single JSON record into a normalized TrainingSample."""
        if fmt == DatasetFormat.ALPACA:
            # Alpaca format: instruction, input (optional), output
            instruction = record.get("instruction", "").strip()
            inp = record.get("input", "").strip()
            output = record.get("output", "").strip()
            if not instruction or not output:
                raise DatasetValidationError("Alpaca records require 'instruction' and 'output' fields.")

            user_content = f"{instruction}\n\nInput: {inp}" if inp else instruction
            return TrainingSample(
                messages=[
                    DialogueTurn(role="user", content=user_content),
                    DialogueTurn(role="assistant", content=output),
                ]
            )

        if fmt == DatasetFormat.SHAREGPT:
            # ShareGPT format: conversations list with from/value or role/content
            convs = record.get("conversations", record.get("items", []))
            if not convs or not isinstance(convs, list):
                raise DatasetValidationError("ShareGPT records require a 'conversations' list.")

            turns: List[DialogueTurn] = []
            for item in convs:
                sender = item.get("from", item.get("role", "")).lower()
                val = item.get("value", item.get("content", "")).strip()

                if sender in ("human", "user"):
                    role = "user"
                elif sender in ("gpt", "assistant", "model", "chatgpt"):
                    role = "assistant"
                elif sender in ("system",):
                    role = "system"
                else:
                    role = "user"

                turns.append(DialogueTurn(role=role, content=val))

            return TrainingSample(messages=turns)

        if fmt in (DatasetFormat.OPENAI_MESSAGES, DatasetFormat.CHATML):
            # OpenAI messages format: list of objects with 'role' and 'content'
            messages = record.get("messages", [])
            if not messages or not isinstance(messages, list):
                raise DatasetValidationError("Record requires a non-empty 'messages' list.")

            turns = []
            for m in messages:
                role = m.get("role", "user").lower()
                content = m.get("content", "").strip()
                turns.append(DialogueTurn(role=role, content=content))

            return TrainingSample(messages=turns)

        if fmt == DatasetFormat.RAW_TEXT:
            text = record.get("text", record.get("content", "")).strip()
            if not text:
                raise DatasetValidationError("Raw text record requires 'text' or 'content' field.")
            return TrainingSample(messages=[DialogueTurn(role="user", content=text)])

        raise DatasetValidationError(f"Unsupported dataset format: {fmt}")

    @classmethod
    def load_jsonl(
        cls,
        path: Path | str,
        fmt: DatasetFormat,
        max_samples: Optional[int] = None,
    ) -> List[TrainingSample]:
        """Load and parse records from a JSONL or JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise DatasetValidationError(f"Dataset file does not exist: {file_path}")

        samples: List[TrainingSample] = []

        # Read JSON file or line-by-line JSONL
        with open(file_path, "r", encoding="utf-8") as f:
            first_char = ""
            while not first_char:
                c = f.read(1)
                if not c:
                    break
                if not c.isspace():
                    first_char = c
            f.seek(0)

            if first_char == "[":
                # Whole JSON array
                raw_data = json.load(f)
                for item in raw_data:
                    samples.append(cls.parse_record(item, fmt))
                    if max_samples and len(samples) >= max_samples:
                        break
            else:
                # JSONL
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    samples.append(cls.parse_record(record, fmt))
                    if max_samples and len(samples) >= max_samples:
                        break

        return samples
