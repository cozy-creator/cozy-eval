"""One device convention for the whole library.

Every backend that places a model takes ``device: str = AUTO``. ``"auto"``
resolves to ``"cuda"`` when a CUDA device is visible and ``"cpu"`` otherwise, so
the same call works on a laptop and on a render pod without the caller threading
a device string through five layers. Any explicit value ("cpu", "cuda",
"cuda:1", "mps") is passed straight through untouched.

Resolving is lazy: a base install with no torch resolves ``"auto"`` to ``"cpu"``
rather than failing to import.
"""

from __future__ import annotations

AUTO = "auto"


def resolve_device(device: str = AUTO) -> str:
    """``"auto"`` -> the best available device; anything else passes through."""
    if device != AUTO:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["AUTO", "resolve_device"]
