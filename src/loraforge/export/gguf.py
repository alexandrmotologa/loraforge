"""GGUF export preparation and llama.cpp quantization command generator.

Prepares conversion commands and shell scripts to quantize merged Safetensors
into GGUF formats (Q4_K_M, Q5_K_M, Q8_0) for lightweight local inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from pydantic import BaseModel


class GGUFQuantizationPlan(BaseModel):
    """Specification of GGUF conversion parameters and target quant level."""

    merged_model_dir: str
    output_gguf_path: str
    quant_method: str = "Q4_K_M"
    llama_cpp_path: str = "llama.cpp"


class GGUFExporter:
    """Prepares scripts and execution instructions for GGUF export."""

    SUPPORTED_QUANTS: List[str] = [
        "Q4_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_0",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "F16",
        "BF16",
    ]

    @classmethod
    def generate_conversion_commands(cls, plan: GGUFQuantizationPlan) -> tuple[str, str]:
        """Generate shell commands to convert HF weights to FP16 GGUF and quantize.

        Returns (convert_command, quantize_command).
        """
        model_dir = Path(plan.merged_model_dir)
        gguf_dir = Path(plan.output_gguf_path).parent
        fp16_gguf = gguf_dir / f"{model_dir.name}-f16.gguf"
        final_gguf = Path(plan.output_gguf_path)

        convert_cmd = (
            f"python {plan.llama_cpp_path}/convert_hf_to_gguf.py "
            f'"{model_dir}" --outfile "{fp16_gguf}" --outtype f16'
        )

        quant_cmd = (
            f'{plan.llama_cpp_path}/llama-quantize '
            f'"{fp16_gguf}" "{final_gguf}" {plan.quant_method}'
        )

        return convert_cmd, quant_cmd

    @classmethod
    def write_conversion_script(cls, plan: GGUFQuantizationPlan, script_path: Path | str) -> Path:
        """Write out an executable conversion script."""
        convert_cmd, quant_cmd = cls.generate_conversion_commands(plan)
        script_file = Path(script_path)
        script_file.parent.mkdir(parents=True, exist_ok=True)

        content = f"""#!/usr/bin/env bash
set -e
echo "[LoRAForge] Step 1: Converting Safetensors to FP16 GGUF..."
{convert_cmd}

echo "[LoRAForge] Step 2: Quantizing to {plan.quant_method}..."
{quant_cmd}

echo "[LoRAForge] GGUF Export completed: {plan.output_gguf_path}"
"""
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(content)

        return script_file
