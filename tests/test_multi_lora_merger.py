"""Mathematical verification and integration tests for Multi-LoRA Adapter Merging (TIES, DARE, SVD, Linear)."""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from loraforge.cli.app import app
from loraforge.core.errors import ExportError
from loraforge.export.merger import (
    AdapterMerger,
    MergeMethod,
    MultiLoraMergeConfig,
    MultiLoraMergeReport,
)
from loraforge.web.app import create_app

try:
    from safetensors.numpy import save_file as save_safetensors
    from safetensors.numpy import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

client = TestClient(create_app())
runner = CliRunner()


def test_linear_merge_math():
    t1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # 1. Default uniform weights
    merged = AdapterMerger.linear_merge([t1, t2])
    expected = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(merged, expected, rtol=1e-5)

    # 2. Custom weights
    merged_weighted = AdapterMerger.linear_merge([t1, t2], weights=[0.2, 0.8])
    expected_weighted = 0.2 * t1 + 0.8 * t2
    np.testing.assert_allclose(merged_weighted, expected_weighted, rtol=1e-5)

    # 3. Error branches
    with pytest.raises(ExportError, match="Cannot merge an empty list"):
        AdapterMerger.linear_merge([])

    with pytest.raises(ExportError, match="shape mismatch"):
        AdapterMerger.linear_merge([t1, np.ones((3, 3), dtype=np.float32)])

    with pytest.raises(ExportError, match="Weights count"):
        AdapterMerger.linear_merge([t1, t2], weights=[0.5])

    with pytest.raises(ExportError, match="Sum of merge weights cannot be zero"):
        AdapterMerger.linear_merge([t1, t2], weights=[0.0, 0.0])


def test_ties_merge_math():
    # Construct tensors where one parameter has conflicting direction
    t1 = np.array([10.0, -5.0, 0.1, 8.0], dtype=np.float32)
    t2 = np.array([8.0, 6.0, 0.2, -1.0], dtype=np.float32)
    t3 = np.array([12.0, -4.0, 0.05, 9.0], dtype=np.float32)

    # With density 0.75, the smallest parameter (0.1, 0.2, 0.05) will be trimmed to 0
    merged = AdapterMerger.ties_merge([t1, t2, t3], density=0.75)
    assert merged.shape == t1.shape

    # First index: all positive (10, 8, 12) -> average is 10.0
    assert abs(merged[0] - 10.0) < 0.1

    # Second index: t1=-5, t2=6, t3=-4. Sum is -3 (negative sign elected).
    # t2 is +6 (disagrees with majority sign, pruned). Average of -5 and -4 is -4.5
    assert abs(merged[1] - (-4.5)) < 0.1

    # Third index: small values, trimmed to zero -> merged is 0
    assert merged[2] == 0.0


def test_dare_merge_math():
    t1 = np.ones((100, 100), dtype=np.float32) * 2.0
    t2 = np.ones((100, 100), dtype=np.float32) * 4.0

    # 1. Zero drop rate -> identical to linear average
    merged_no_drop = AdapterMerger.dare_merge([t1, t2], drop_rate=0.0)
    np.testing.assert_allclose(merged_no_drop, np.ones((100, 100)) * 3.0)

    # 2. Stochastic drop rate 0.5 -> expected value should be approximately 3.0
    merged_dare = AdapterMerger.dare_merge([t1, t2], drop_rate=0.5, seed=42)
    # Mean of rescaled values should match expectation within standard error
    assert abs(np.mean(merged_dare) - 3.0) < 0.15


def test_svd_truncate_math():
    rng = np.random.default_rng(42)
    # Construct exact rank-2 matrix of dimension (16, 16)
    B_true = rng.standard_normal((16, 2)).astype(np.float32)
    A_true = rng.standard_normal((2, 16)).astype(np.float32)
    M = B_true @ A_true

    # 1. Truncate to rank 2 -> preserves 100% variance, zero reconstruction error
    b_mat, a_mat, var_ratio, err = AdapterMerger.svd_truncate(M, target_rank=2)
    assert b_mat.shape == (16, 2)
    assert a_mat.shape == (2, 16)
    assert var_ratio == 1.0
    assert err < 1e-4

    # 2. Truncate to rank 1 -> preserves leading component
    b_mat_1, a_mat_1, var_ratio_1, err_1 = AdapterMerger.svd_truncate(M, target_rank=1)
    assert b_mat_1.shape == (16, 1)
    assert a_mat_1.shape == (1, 16)
    assert 0.0 < var_ratio_1 < 1.0
    assert err_1 > 0.0

    # 3. Non-2D matrix error
    with pytest.raises(ExportError, match="requires 2D matrix"):
        AdapterMerger.svd_truncate(np.ones((4, 4, 4)), target_rank=2)


def test_merge_weight_dicts_all_methods():
    d1 = {
        "base_model.model.q_proj.weight": np.ones((8, 8), dtype=np.float32) * 1.0,
        "base_model.model.v_proj.weight": np.ones((8, 8), dtype=np.float32) * 2.0,
    }
    d2 = {
        "base_model.model.q_proj.weight": np.ones((8, 8), dtype=np.float32) * 3.0,
        "base_model.model.v_proj.weight": np.ones((8, 8), dtype=np.float32) * 4.0,
    }

    # Test each method
    for method in (MergeMethod.LINEAR, MergeMethod.TIES, MergeMethod.DARE, MergeMethod.SVD):
        cfg = MultiLoraMergeConfig(
            method=method,
            density=0.5,
            drop_rate=0.3,
            target_rank=4,
        )
        merged_dict, report = AdapterMerger.merge_weight_dicts([d1, d2], cfg)
        assert len(merged_dict) == 2
        assert report.tensors_merged == 2
        assert report.total_parameters == 128
        assert "base_model.model.q_proj.weight" in merged_dict


def test_merge_adapters_from_disk_and_cli(tmp_path: Path):
    if not HAS_SAFETENSORS:
        pytest.skip("Safetensors not available in environment.")

    adapter1_dir = tmp_path / "adapter_alpha"
    adapter2_dir = tmp_path / "adapter_beta"
    out_dir = tmp_path / "merged_out"
    adapter1_dir.mkdir()
    adapter2_dir.mkdir()

    weights1 = {"lora_A": np.ones((4, 8), dtype=np.float32), "lora_B": np.ones((8, 4), dtype=np.float32)}
    weights2 = {"lora_A": np.ones((4, 8), dtype=np.float32) * 2.0, "lora_B": np.ones((8, 4), dtype=np.float32) * 3.0}

    save_safetensors(weights1, str(adapter1_dir / "adapter_model.safetensors"))
    save_safetensors(weights2, str(adapter2_dir / "adapter_model.safetensors"))

    # 1. Direct Python API merge
    cfg = MultiLoraMergeConfig(
        method=MergeMethod.TIES,
        output_dir=str(out_dir),
        density=0.8,
    )
    report = AdapterMerger.merge_adapters_from_files([adapter1_dir, adapter2_dir], cfg)
    assert report.num_adapters == 2
    assert (out_dir / "adapter_model.safetensors").is_file()
    assert (out_dir / "merge_report.json").is_file()

    # Verify loaded content
    reloaded = load_safetensors(str(out_dir / "adapter_model.safetensors"))
    assert "lora_A" in reloaded
    assert reloaded["lora_A"].shape == (4, 8)

    # 2. CLI merge invocation
    cli_out = tmp_path / "cli_merged"
    cli_res = runner.invoke(app, [
        "merge",
        "--adapters", f"{adapter1_dir},{adapter2_dir}",
        "--out", str(cli_out),
        "--method", "ties",
        "--density", "0.5",
    ])
    assert cli_res.exit_code == 0
    assert "Multi-LoRA Adapter Consolidation Summary" in cli_res.output
    assert (cli_out / "adapter_model.safetensors").is_file()


def test_api_export_merge(tmp_path: Path):
    if not HAS_SAFETENSORS:
        pytest.skip("Safetensors not available.")

    ad1 = tmp_path / "api_ad1"
    ad2 = tmp_path / "api_ad2"
    ad1.mkdir()
    ad2.mkdir()
    save_safetensors({"w": np.ones((2, 2), dtype=np.float32)}, str(ad1 / "adapter_model.safetensors"))
    save_safetensors({"w": np.ones((2, 2), dtype=np.float32) * 3.0}, str(ad2 / "adapter_model.safetensors"))

    payload = {
        "adapter_paths": [str(ad1), str(ad2)],
        "output_dir": str(tmp_path / "api_merged_out"),
        "method": "ties",
        "density": 0.8,
    }
    res = client.post("/api/export/merge", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "ties"
    assert data["num_adapters"] == 2
    assert data["tensors_merged"] == 1
    assert "adapter_model.safetensors" in data["output_file"]
