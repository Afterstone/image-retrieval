"""Local backend selection; inference never silently retries on CPU."""


def get_device(selection: str = "auto") -> str:
    import os

    import torch

    if selection not in {"auto", "cpu", "mps"}:
        raise ValueError("Device must be auto, mps, or cpu")
    available = torch.backends.mps.is_available()
    if selection == "mps" and not available:
        raise ValueError("MPS was requested but is unavailable")
    resolved = "mps" if available and selection != "cpu" else "cpu"
    if resolved == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError(
            "Disable PYTORCH_ENABLE_MPS_FALLBACK: silent CPU fallback is unsupported"
        )
    return resolved
