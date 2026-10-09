"""Unit tests for InferOps manifest exporter, merge plans, and GGUF quantization."""

from pathlib import Path
from loraforge.core.config import DatasetConfig, FineTuningJobConfig
from loraforge.export.gguf import GGUFExporter, GGUFQuantizationPlan
from loraforge.export.inferops import InferOpsExporter
from loraforge.export.merger import ModelMerger


def test_inferops_manifest_export(tmp_path, llama3_8b_spec, rtx_3060_spec):
    cfg = FineTuningJobConfig(
        job_name="agent-llama3",
        model=llama3_8b_spec,
        gpu=rtx_3060_spec,
        dataset=DatasetConfig(path="train.jsonl"),
    )

    out_dir = tmp_path / "exports"
    manifest_path = InferOpsExporter.export_bundle(cfg, out_dir, engine="vllm")

    assert manifest_path.is_file()
    assert (out_dir / "inferops_manifest.json").is_file()
    assert (out_dir / "serve_in_inferops.sh").is_file()
    assert (out_dir / "serve_in_inferops.bat").is_file()

    content = manifest_path.read_text(encoding="utf-8")
    assert "agent-llama3" in content
    assert "engine: vllm" in content
    assert "meta-llama/Llama-3.1-8B-Instruct" in content


def test_gguf_conversion_plan():
    plan = GGUFQuantizationPlan(
        merged_model_dir="./merged_model",
        output_gguf_path="./models/model-q4.gguf",
        quant_method="Q4_K_M",
    )
    convert_cmd, quant_cmd = GGUFExporter.generate_conversion_commands(plan)

    assert "convert_hf_to_gguf.py" in convert_cmd
    assert "llama-quantize" in quant_cmd
    assert "Q4_K_M" in quant_cmd


def test_merge_plan_creation():
    plan = ModelMerger.create_merge_plan(
        base_model_path="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path="./checkpoints/adapter",
        output_dir="./merged",
        target_dtype="bfloat16",
    )
    assert plan.target_dtype == "bfloat16"
    assert plan.safe_serialization is True
