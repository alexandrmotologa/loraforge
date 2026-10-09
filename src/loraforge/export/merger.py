"""LoRA adapter merging, multi-adapter consolidation, and rank compression.

Provides mathematically sound fusion of multiple low-rank adapters:
- Linear weighted interpolation
- TIES merging (Trimming, Electing Sign, Disjoint Merge - Yadav et al. 2023)
- DARE (Drop And REscale - Yu et al. 2023)
- SVD rank truncation & variance-preserving compression
- 16-bit Safetensors serialization
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from loraforge.core.errors import ExportError

try:
    from safetensors.numpy import load_file as load_safetensors_file
    from safetensors.numpy import save_file as save_safetensors_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class MergeMethod(str, Enum):
    """Algorithms for blending multiple LoRA parameter matrices."""

    LINEAR = "linear"
    TIES = "ties"
    DARE = "dare"
    SVD = "svd"


class MultiLoraMergeConfig(BaseModel):
    """Configuration parameters for multi-adapter consolidation."""

    method: MergeMethod = MergeMethod.TIES
    adapter_paths: List[str] = Field(default_factory=list)
    weights: Optional[List[float]] = None
    density: float = Field(0.2, ge=0.01, le=1.0, description="Top-k density fraction kept for TIES trimming.")
    drop_rate: float = Field(0.5, ge=0.0, lt=1.0, description="Bernoulli drop probability for DARE.")
    target_rank: Optional[int] = Field(None, gt=0, description="Target rank for SVD truncation compression.")
    output_dir: str = "./merged_adapters"
    safe_serialization: bool = True


class MultiLoraMergeReport(BaseModel):
    """Statistical verification report for an adapter merge execution."""

    method: MergeMethod
    num_adapters: int
    tensors_merged: int
    total_parameters: int
    preserved_variance: Optional[float] = None
    density: float
    output_file: str
    summary: str


class AdapterMerger:
    """Core mathematical engine for multi-LoRA merging and SVD rank truncation."""

    @staticmethod
    def linear_merge(
        tensors: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Weighted linear sum of adapter delta tensors: sum(w_i * T_i)."""
        if not tensors:
            raise ExportError("Cannot merge an empty list of tensors.")

        first_shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], 1):
            if t.shape != first_shape:
                raise ExportError(f"Tensor shape mismatch at index {i}: {t.shape} vs {first_shape}.")

        num = len(tensors)
        if weights is None:
            w = [1.0 / num] * num
        else:
            if len(weights) != num:
                raise ExportError(f"Weights count ({len(weights)}) does not match tensors count ({num}).")
            total_w = sum(weights)
            if total_w == 0:
                raise ExportError("Sum of merge weights cannot be zero.")
            w = [val / total_w for val in weights]

        result = np.zeros_like(tensors[0], dtype=np.float32)
        for weight, t in zip(w, tensors):
            result += weight * t.astype(np.float32)

        return result.astype(tensors[0].dtype)

    @staticmethod
    def ties_merge(
        tensors: List[np.ndarray],
        density: float = 0.2,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """TIES Merging: Trimming (density), Electing Sign, and Disjoint Average.

        Reference: Yadav et al. "Resolving Interference When Merging Models" (NeurIPS 2023).
        """
        if not tensors:
            raise ExportError("Cannot merge an empty list of tensors.")

        num = len(tensors)
        first_shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], 1):
            if t.shape != first_shape:
                raise ExportError(f"Tensor shape mismatch at index {i}: {t.shape} vs {first_shape}.")

        if weights is None:
            w = [1.0 / num] * num
        else:
            if len(weights) != num:
                raise ExportError(f"Weights count ({len(weights)}) does not match tensors count ({num}).")
            total_w = sum(weights)
            w = [val / total_w for val in weights] if total_w != 0 else [1.0 / num] * num

        # Step 1: Trim (keep top density fraction by magnitude per tensor)
        trimmed_tensors: List[np.ndarray] = []
        for t in tensors:
            flat = np.abs(t).flatten()
            if density < 1.0:
                k = int(len(flat) * (1.0 - density))
                if k > 0 and k < len(flat):
                    threshold = np.partition(flat, k)[k]
                    trimmed = np.where(np.abs(t) >= threshold, t.astype(np.float32), 0.0)
                else:
                    trimmed = t.astype(np.float32)
            else:
                trimmed = t.astype(np.float32)
            trimmed_tensors.append(trimmed)

        # Step 2: Elect Sign
        weighted_sum = np.zeros_like(tensors[0], dtype=np.float32)
        for weight, t_trim in zip(w, trimmed_tensors):
            weighted_sum += weight * t_trim

        elected_sign = np.sign(weighted_sum)

        # Step 3: Disjoint Average (average only parameters that agree with the majority sign)
        disjoint_sum = np.zeros_like(tensors[0], dtype=np.float32)
        disjoint_counts = np.zeros_like(tensors[0], dtype=np.float32)

        for weight, t_trim in zip(w, trimmed_tensors):
            signs = np.sign(t_trim)
            matches = (signs == elected_sign) & (t_trim != 0)
            disjoint_sum += np.where(matches, weight * t_trim, 0.0)
            disjoint_counts += np.where(matches, weight, 0.0)

        # Avoid division by zero where no parameters matched
        merged = np.divide(
            disjoint_sum,
            disjoint_counts,
            out=np.zeros_like(disjoint_sum),
            where=disjoint_counts > 0,
        )
        return merged.astype(tensors[0].dtype)

    @staticmethod
    def dare_merge(
        tensors: List[np.ndarray],
        drop_rate: float = 0.5,
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """DARE (Drop And REscale) Merging with stochastic Bernoulli masking.

        Reference: Yu et al. "Language Models are Super Mario: Absorbing Abilities from Homologous Models" (2023).
        """
        if not tensors:
            raise ExportError("Cannot merge an empty list of tensors.")

        num = len(tensors)
        first_shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], 1):
            if t.shape != first_shape:
                raise ExportError(f"Tensor shape mismatch at index {i}: {t.shape} vs {first_shape}.")

        if weights is None:
            w = [1.0 / num] * num
        else:
            if len(weights) != num:
                raise ExportError(f"Weights count ({len(weights)}) does not match tensors count ({num}).")
            total_w = sum(weights)
            w = [val / total_w for val in weights] if total_w != 0 else [1.0 / num] * num

        rng = np.random.default_rng(seed)
        rescaled_tensors: List[np.ndarray] = []

        scale_factor = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0

        for t in tensors:
            if drop_rate > 0.0:
                mask = rng.binomial(1, 1.0 - drop_rate, size=t.shape).astype(np.float32)
                rescaled = (t.astype(np.float32) * mask) * scale_factor
            else:
                rescaled = t.astype(np.float32)
            rescaled_tensors.append(rescaled)

        result = np.zeros_like(tensors[0], dtype=np.float32)
        for weight, r_tensor in zip(w, rescaled_tensors):
            result += weight * r_tensor

        return result.astype(tensors[0].dtype)

    @staticmethod
    def svd_truncate(
        matrix: np.ndarray,
        target_rank: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Truncate weight matrix into low-rank factors B and A via Singular Value Decomposition.

        Returns:
            Tuple of:
            - B matrix (d_out, target_rank)
            - A matrix (target_rank, d_in)
            - preserved_variance_ratio (float)
            - frobenius_reconstruction_error (float)
        """
        if matrix.ndim != 2:
            raise ExportError(f"SVD truncation requires 2D matrix, received shape {matrix.shape}.")

        u, s, vh = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)

        k = min(target_rank, len(s))
        u_k = u[:, :k]
        s_k = s[:k]
        vh_k = vh[:k, :]

        # Factorize singular values evenly across B and A: B = U * sqrt(S), A = sqrt(S) * V^T
        sqrt_s = np.sqrt(s_k)
        b = u_k * sqrt_s[None, :]
        a = sqrt_s[:, None] * vh_k

        # Reconstructed matrix
        reconstructed = b @ a

        total_variance = float(np.sum(s ** 2))
        preserved_variance = float(np.sum(s_k ** 2))
        variance_ratio = round((preserved_variance / total_variance), 4) if total_variance > 0 else 1.0
        frobenius_error = float(np.linalg.norm(matrix.astype(np.float32) - reconstructed))

        return (
            b.astype(matrix.dtype),
            a.astype(matrix.dtype),
            variance_ratio,
            frobenius_error,
        )

    @classmethod
    def merge_weight_dicts(
        cls,
        adapter_dicts: List[Dict[str, np.ndarray]],
        config: MultiLoraMergeConfig,
    ) -> Tuple[Dict[str, np.ndarray], MultiLoraMergeReport]:
        """Merge multiple state dictionaries according to configured algorithm."""
        if not adapter_dicts:
            raise ExportError("No adapter state dictionaries provided for merging.")

        # Find intersecting tensor keys
        common_keys = set(adapter_dicts[0].keys())
        for d in adapter_dicts[1:]:
            common_keys &= set(d.keys())

        if not common_keys:
            raise ExportError("No overlapping parameter tensor keys found between adapters.")

        merged_dict: Dict[str, np.ndarray] = {}
        total_params = 0
        preserved_variances: List[float] = []

        for key in sorted(common_keys):
            tensors = [d[key] for d in adapter_dicts]
            total_params += tensors[0].size

            if config.method == MergeMethod.LINEAR:
                merged = cls.linear_merge(tensors, weights=config.weights)
            elif config.method == MergeMethod.TIES:
                merged = cls.ties_merge(tensors, density=config.density, weights=config.weights)
            elif config.method == MergeMethod.DARE:
                merged = cls.dare_merge(tensors, drop_rate=config.drop_rate, weights=config.weights)
            elif config.method == MergeMethod.SVD:
                # First linear blend then SVD compress if rank is requested
                linear_base = cls.linear_merge(tensors, weights=config.weights)
                if config.target_rank and linear_base.ndim == 2 and linear_base.shape[0] > config.target_rank:
                    b_mat, a_mat, var_ratio, _ = cls.svd_truncate(linear_base, config.target_rank)
                    merged = b_mat @ a_mat
                    preserved_variances.append(var_ratio)
                else:
                    merged = linear_base
            else:
                raise ExportError(f"Unsupported merge method: {config.method}")

            merged_dict[key] = merged

        mean_variance = float(np.mean(preserved_variances)) if preserved_variances else None

        report = MultiLoraMergeReport(
            method=config.method,
            num_adapters=len(adapter_dicts),
            tensors_merged=len(merged_dict),
            total_parameters=total_params,
            preserved_variance=mean_variance,
            density=config.density if config.method == MergeMethod.TIES else (1.0 - config.drop_rate if config.method == MergeMethod.DARE else 1.0),
            output_file=str(Path(config.output_dir) / "adapter_model.safetensors"),
            summary=(
                f"Successfully merged {len(adapter_dicts)} adapters using {config.method.value.upper()} "
                f"across {len(merged_dict)} parameter tensors ({total_params:,} parameters)."
            ),
        )

        return merged_dict, report

    @classmethod
    def merge_adapters_from_files(
        cls,
        adapter_paths: List[Path | str],
        config: MultiLoraMergeConfig,
    ) -> MultiLoraMergeReport:
        """Load adapter safetensors files from disk, merge them, and save result."""
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        dicts: List[Dict[str, np.ndarray]] = []
        for p in adapter_paths:
            path = Path(p)
            file_to_load = path
            if path.is_dir():
                candidate = path / "adapter_model.safetensors"
                if candidate.is_file():
                    file_to_load = candidate
                else:
                    raise ExportError(f"Directory {path} does not contain adapter_model.safetensors.")

            if not file_to_load.is_file():
                raise ExportError(f"Adapter file not found: {file_to_load}")

            if HAS_SAFETENSORS and file_to_load.suffix == ".safetensors":
                tensors = load_safetensors_file(str(file_to_load))
            else:
                # NumPy npz fallback
                data = np.load(str(file_to_load))
                tensors = {k: data[k] for k in data.files}
            dicts.append(tensors)

        merged_dict, report = cls.merge_weight_dicts(dicts, config)

        target_file = out_path / "adapter_model.safetensors"
        if HAS_SAFETENSORS:
            save_safetensors_file(merged_dict, str(target_file))
        else:
            np.savez(str(out_path / "adapter_model.npz"), **merged_dict)

        # Write metadata manifest
        meta = {
            "method": config.method.value,
            "adapters": [str(p) for p in adapter_paths],
            "tensors_merged": report.tensors_merged,
            "total_parameters": report.total_parameters,
            "preserved_variance": report.preserved_variance,
            "density": report.density,
        }
        with open(out_path / "merge_report.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return report


class MergePlan(BaseModel):
    """Execution blueprint for base model checkpoint merging."""

    base_model_path: str
    adapter_path: str
    output_dir: str
    target_dtype: str = "bfloat16"
    safe_serialization: bool = True


class ModelMerger:
    """Consolidates LoRA weights with base model checkpoints into 16-bit checkpoints."""

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
        """Execute adapter weight merging."""
        out_dir = Path(plan.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
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
