"""Custom exceptions for LoRAForge."""


class LoRAForgeError(Exception):
    """Base exception for all LoRAForge domain errors."""


class HardwareIncompatibleError(LoRAForgeError):
    """Raised when hardware requirements for a training run cannot be met."""


class OOMRiskError(LoRAForgeError):
    """Raised when predictive VRAM allocation exceeds available or safe GPU memory."""

    def __init__(
        self,
        predicted_mb: float,
        available_mb: float,
        safe_threshold_mb: float,
        message: str = "",
    ):
        self.predicted_mb = predicted_mb
        self.available_mb = available_mb
        self.safe_threshold_mb = safe_threshold_mb
        detail = (
            message
            or f"Predicted peak VRAM ({predicted_mb:.1f} MB) exceeds safe budget "
            f"({safe_threshold_mb:.1f} MB / available {available_mb:.1f} MB)."
        )
        super().__init__(detail)


class DatasetValidationError(LoRAForgeError):
    """Raised when a dataset fails schema, tokenization, or packing validation."""


class SupervisorError(LoRAForgeError):
    """Raised when the training process supervisor encounters an unrecoverable failure."""


class ExportError(LoRAForgeError):
    """Raised when adapter merging, GGUF conversion, or InferOps export fails."""
