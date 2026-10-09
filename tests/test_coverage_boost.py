"""Additional unit tests to boost test coverage across edge cases and helper methods."""

import json
from pathlib import Path
import pytest

from loraforge.core.config import (
    DatasetConfig,
    DatasetFormat,
    FineTuningJobConfig,
    LoRAConfig,
    OptimizerType,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)
from loraforge.core.errors import (
    DatasetValidationError,
    ExportError,
    HardwareIncompatibleError,
    LoRAForgeError,
    OOMRiskError,
    SupervisorError,
)
from loraforge.core.models import (
    PRESET_MODELS,
    get_model_spec,
    parse_huggingface_config,
)
from loraforge.data.formats import DatasetParser
from loraforge.export.gguf import GGUFExporter, GGUFQuantizationPlan
from loraforge.export.merger import MergePlan, ModelMerger
from loraforge.supervisor.runner import JobStatus, TrainingSupervisor
from loraforge.vram.calculator import VRAMCalculator


def test_custom_errors_representation():
    err = OOMRiskError(predicted_mb=14000.0, available_mb=12288.0, safe_threshold_mb=11500.0)
    assert "14000.0 MB" in str(err)

    err_custom = OOMRiskError(14000, 12000, 11000, message="Custom OOM detail")
    assert str(err_custom) == "Custom OOM detail"

    assert issubclass(HardwareIncompatibleError, LoRAForgeError)
    assert issubclass(DatasetValidationError, LoRAForgeError)
    assert issubclass(SupervisorError, LoRAForgeError)
    assert issubclass(ExportError, LoRAForgeError)


def test_models_lookup_and_config_parser(tmp_path):
    # Lookup by short name
    spec = get_model_spec("llama-3.1-8b-instruct")
    assert spec is not None
    assert spec.num_layers == 32

    # Lookup by full name
    spec2 = get_model_spec("Qwen/Qwen2.5-7B-Instruct")
    assert spec2 is not None

    # Unknown
    assert get_model_spec("completely_fake_model") is None

    # Parse HuggingFace config.json
    hf_cfg = {
        "_name_or_path": "custom/test-model",
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
    }
    cfg_file = tmp_path / "config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f)

    parsed = parse_huggingface_config(cfg_file)
    assert parsed.name == "custom/test-model"
    assert parsed.num_layers == 24
    assert parsed.hidden_size == 2048
    assert parsed.num_parameters > 0

    # Also test get_model_spec on directory
    spec_dir = get_model_spec(str(tmp_path))
    assert spec_dir is not None
    assert spec_dir.num_layers == 24


def test_formats_load_jsonl_and_errors(tmp_path):
    # Test valid JSONL loading
    lines = [
        json.dumps({"instruction": "Task 1", "output": "Done 1"}),
        json.dumps({"instruction": "Task 2", "output": "Done 2"}),
    ]
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    samples = DatasetParser.load_jsonl(jsonl_file, DatasetFormat.ALPACA, max_samples=1)
    assert len(samples) == 1
    assert samples[0].messages[0].content == "Task 1"

    # Test valid JSON array loading
    json_arr = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    arr_file = tmp_path / "arr.json"
    with open(arr_file, "w", encoding="utf-8") as f:
        json.dump([{"messages": json_arr}], f)

    arr_samples = DatasetParser.load_jsonl(arr_file, DatasetFormat.OPENAI_MESSAGES)
    assert len(arr_samples) == 1

    # Test Raw Text parsing
    raw_sample = DatasetParser.parse_record({"text": "Just plain prose."}, DatasetFormat.RAW_TEXT)
    assert raw_sample.messages[0].content == "Just plain prose."

    # Test Missing File error
    with pytest.raises(DatasetValidationError):
        DatasetParser.load_jsonl(tmp_path / "non_existent.jsonl", DatasetFormat.ALPACA)

    # Test Record errors
    with pytest.raises(DatasetValidationError):
        DatasetParser.parse_record({"bad": "data"}, DatasetFormat.ALPACA)

    with pytest.raises(DatasetValidationError):
        DatasetParser.parse_record({"bad": "data"}, DatasetFormat.SHAREGPT)

    with pytest.raises(DatasetValidationError):
        DatasetParser.parse_record({"bad": "data"}, DatasetFormat.OPENAI_MESSAGES)

    with pytest.raises(DatasetValidationError):
        DatasetParser.parse_record({"bad": "data"}, DatasetFormat.RAW_TEXT)


def test_vram_calculator_additional_options(llama3_8b_spec, rtx_3060_spec):
    calc = VRAMCalculator()

    # Modules to save (e.g. unfreezing lm_head)
    lora_cfg = LoRAConfig(r=8, target_modules=["q_proj"], modules_to_save=["lm_head"])
    bd = calc.calculate_lora_parameters(llama3_8b_spec, lora_cfg)
    assert bd.trainable_parameters > 0

    # INT8 Quantization
    w_int8, meta_int8 = calc.calculate_base_model_memory(
        llama3_8b_spec,
        QuantizationConfig(quant_type=QuantizationType.INT8),
    )
    assert w_int8 > 0
    assert meta_int8 > 0

    # FP4 Quantization
    w_fp4, meta_fp4 = calc.calculate_base_model_memory(
        llama3_8b_spec,
        QuantizationConfig(quant_type=QuantizationType.FP4, double_quant=False),
    )
    assert w_fp4 > 0

    # Quantize lm head = True
    w_head_q, meta_head_q = calc.calculate_base_model_memory(
        llama3_8b_spec,
        QuantizationConfig(quant_type=QuantizationType.NF4, quantize_lm_head=True),
    )
    assert w_head_q < w_int8

    # Lion 8-bit optimizer memory
    lion_mem = calc.calculate_optimizer_memory(1_000_000, OptimizerType.LION_8BIT)
    assert lion_mem > 0

    # SGD optimizer memory
    sgd_mem = calc.calculate_optimizer_memory(1_000_000, OptimizerType.SGD)
    assert sgd_mem > 0

    # Paged 32-bit optimizer
    p32_mem = calc.calculate_optimizer_memory(1_000_000, OptimizerType.PAGED_ADAMW_32BIT)
    assert p32_mem > lion_mem


def test_supervisor_worker_generation_and_cancellation(tmp_path, llama3_8b_spec, rtx_3060_spec):
    supervisor = TrainingSupervisor(work_dir=str(tmp_path))

    cfg = FineTuningJobConfig(
        job_name="worker-test",
        model=llama3_8b_spec,
        gpu=rtx_3060_spec,
        dataset=DatasetConfig(path="train.jsonl"),
    )

    worker_file = tmp_path / "worker.py"
    supervisor.generate_worker_script(cfg, worker_file)
    assert worker_file.is_file()
    assert "LoRAForge Supervisor" in worker_file.read_text(encoding="utf-8")

    # Test cancellation logic
    job_id = "test-job-to-cancel"
    supervisor._stop_flags[job_id] = False
    cancelled = supervisor.cancel_job(job_id)
    assert cancelled is False  # No active subprocess


def test_merger_and_gguf_scripts(tmp_path):
    # Test merger fallback when torch not loaded
    plan = ModelMerger.create_merge_plan(
        base_model_path="dummy-base",
        adapter_path="dummy-adapter",
        output_dir=str(tmp_path / "merged_out"),
    )
    out_dir = ModelMerger.execute_merge(plan)
    assert out_dir.is_dir()
    assert (out_dir / "merge_manifest.json").is_file()

    # Test GGUF script generation
    gguf_plan = GGUFQuantizationPlan(
        merged_model_dir=str(out_dir),
        output_gguf_path=str(tmp_path / "out.gguf"),
        quant_method="Q8_0",
    )
    script_path = GGUFExporter.write_conversion_script(gguf_plan, tmp_path / "convert.sh")
    assert script_path.is_file()
    script_text = script_path.read_text(encoding="utf-8")
    assert "Q8_0" in script_text
    assert "convert_hf_to_gguf.py" in script_text


def test_formats_edge_cases_and_parsers(tmp_path):
    # 1. Raw text record
    raw_sample = DatasetParser.parse_record({"text": "This is raw unstructured text."}, DatasetFormat.RAW_TEXT)
    assert len(raw_sample.messages) == 1
    assert raw_sample.messages[0].role == "user"

    # Test prompt formatting fallback
    prompt_fb = raw_sample.to_formatted_prompt("unknown_template")
    assert "### User:" in prompt_fb

    prompt_llama = raw_sample.to_formatted_prompt("llama3")
    assert "<|begin_of_text|>" in prompt_llama

    # 2. OpenAI messages record
    oai_sample = DatasetParser.parse_record({
        "messages": [
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": "Pong"},
        ]
    }, DatasetFormat.OPENAI_MESSAGES)
    assert len(oai_sample.messages) == 2

    # 3. JSON array file loading
    json_array_file = tmp_path / "array.json"
    json_array_file.write_text(json.dumps([{"instruction": "A", "output": "B"}]), encoding="utf-8")
    array_samples = DatasetParser.load_jsonl(json_array_file, DatasetFormat.ALPACA)
    assert len(array_samples) == 1

    # 4. Invalid JSON line handling
    corrupt_file = tmp_path / "corrupt.jsonl"
    corrupt_file.write_text("not a json\n{\"instruction\": \"Valid\", \"output\": \"1\"}\n", encoding="utf-8")
    with pytest.raises(Exception):
        DatasetParser.load_jsonl(corrupt_file, DatasetFormat.ALPACA)


def test_optimizer_ultra_constrained_budget(llama3_8b_spec):
    from loraforge.core.config import GPUSpec
    from loraforge.vram.optimizer import HyperparameterOptimizer

    tiny_gpu = GPUSpec(
        name="Tiny 2GB GPU",
        total_vram_mb=2048,
        usable_vram_mb=1800,
        compute_capability_major=8,
        compute_capability_minor=6,
    )
    opt = HyperparameterOptimizer()
    rec = opt.auto_tune_for_gpu(llama3_8b_spec, tiny_gpu, target_effective_batch_size=4, target_seq_len=2048)
    assert rec.headroom_mb < 0  # Severe OOM
    assert rec.is_safe is False
    assert rec.optimal_micro_batch_size == 1



def test_merger_edge_cases(tmp_path):
    import numpy as np
    from loraforge.export.merger import AdapterMerger, MergeMethod, MultiLoraMergeConfig

    t1 = np.ones((4, 4), dtype=np.float32)
    t2 = np.ones((4, 4), dtype=np.float32) * 2.0

    # TIES with density 1.0 and custom weights
    merged_ties = AdapterMerger.ties_merge([t1, t2], density=1.0, weights=[0.4, 0.6])
    assert merged_ties.shape == (4, 4)

    # TIES weights mismatch
    with pytest.raises(ExportError, match="Weights count"):
        AdapterMerger.ties_merge([t1, t2], weights=[1.0])

    # DARE with custom weights
    merged_dare = AdapterMerger.dare_merge([t1, t2], drop_rate=0.0, weights=[0.5, 0.5])
    assert merged_dare.shape == (4, 4)

    # DARE weights mismatch
    with pytest.raises(ExportError, match="Weights count"):
        AdapterMerger.dare_merge([t1, t2], weights=[1.0])

    # Empty weight dicts
    with pytest.raises(ExportError, match="No adapter state dictionaries"):
        AdapterMerger.merge_weight_dicts([], MultiLoraMergeConfig())

    # No overlapping keys
    with pytest.raises(ExportError, match="No overlapping parameter tensor"):
        AdapterMerger.merge_weight_dicts([{"a": t1}, {"b": t2}], MultiLoraMergeConfig())

    # Adapter file not found
    with pytest.raises(ExportError, match="not found"):
        AdapterMerger.merge_adapters_from_files(["non_existent_adapter_file.safetensors"], MultiLoraMergeConfig())

    # Adapter dir missing safetensors
    empty_dir = tmp_path / "empty_adapter_dir"
    empty_dir.mkdir()
    with pytest.raises(ExportError, match="does not contain adapter_model.safetensors"):
        AdapterMerger.merge_adapters_from_files([empty_dir], MultiLoraMergeConfig())


def test_cli_error_handling(tmp_path):
    from typer.testing import CliRunner
    from loraforge.cli.app import app

    runner = CliRunner()

    # 1. autotune unknown gpu
    res1 = runner.invoke(app, ["autotune", "--model", "meta-llama/Llama-3.1-8B-Instruct", "--gpu", "fake_gpu"])
    assert res1.exit_code != 0
    assert "Unknown GPU" in res1.output

    # 2. data audit non-existent file
    res2 = runner.invoke(app, ["data", "audit", "non_existent_file.jsonl"])
    assert res2.exit_code != 0
    assert "File not found" in res2.output

    # 3. data clean non-existent file
    res3 = runner.invoke(app, ["data", "clean", "non_existent.jsonl", "-o", "out.jsonl"])
    assert res3.exit_code != 0
    assert "File not found" in res3.output

    # 4. merge empty adapters
    res5 = runner.invoke(app, ["merge", "--adapters", ""])
    assert res5.exit_code != 0

    # 5. merge invalid weights
    res6 = runner.invoke(app, ["merge", "--adapters", "a,b", "--weights", "not_numbers"])
    assert res6.exit_code != 0

    # 6. merge invalid method
    res7 = runner.invoke(app, ["merge", "--adapters", "a,b", "--method", "unsupported_method"])
    assert res7.exit_code != 0

