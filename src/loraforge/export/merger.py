"""LoRA adapter merging and full checkpoint consolidation.

Merges low-rank delta weights into base model parameters and writes
consolidated 16-bit Safetensors checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

from loraforge.core.errors import ExportError


class MergePlan(BaseModel):
    """Execution blueprint for adapter merging."""

    base_model_path: str
    adapter_path: str
    output_dir: str
    target_dtype: str = "bfloat16"
    safe_serialization: bool = True


class ModelMerger:
    """Consolidates LoRA weights with base model checkpoints."""

    @staticmethod
    def create_merge_plan(
        base_model_path: str,
        adapter_path: str,
        output_dir: str,
        target_dtype: str = "bfloat16",
    ) -> MergePlan:
        """Construct a validated plan for merging adapter weights."""
        return MergePlan(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_dir=output_dir,
            target_dtype=target_dtype,
            safe_serialization=True,
        )

    @classmethod
    def execute_merge(cls, plan: MergePlan) -> Path:
        """Execute adapter weight merging.

        Uses peft.AutoPeftModelForCausalLM if available; writes consolidated
        descriptor artifacts.
        """
        out_dir = Path(plan.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Check for installed PyTorch and PEFT
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"[LoRAForge] Loading base model '{plan.base_model_path}'...")
            torch_dtype = torch.bfloat16 if plan.target_dtype == "bfloat16" else torch.float16
            base_model = AutoModelForCausalLM.from_pretrained(
                plan.base_model_path,
                torch_dtype=torch_dtype,
                device_map="cpu",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(plan.base_model_path)

            print(f"[LoRAForge] Attaching adapter from '{plan.adapter_path}'...")
            peft_model = PeftModel.from_pretrained(base_model, plan.adapter_path)

            print("[LoRAForge] Merging low-rank delta weights into base matrices...")
            merged_model = peft_model.merge_and_unload()

            print(f"[LoRAForge] Saving merged weights to '{plan.output_dir}'...")
            merged_model.save_pretrained(out_dir, safe_serialization=plan.safe_serialization)
            tokenizer.save_pretrained(out_dir)

        except ImportError:
            # If running in environment without torch/transformers/peft installed,
            # write structural metadata and instructions.
            meta = {
                "status": "manifest_only",
                "base_model": plan.base_model_path,
                "adapter": plan.adapter_path,
                "target_dtype": plan.target_dtype,
                "merge_command": (
                    f"python -m loraforge.export.merge_cli "
                    f"--base {plan.base_model_path} --adapter {plan.adapter_path} --out {plan.output_dir}"
                ),
            }
            with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return out_dir
