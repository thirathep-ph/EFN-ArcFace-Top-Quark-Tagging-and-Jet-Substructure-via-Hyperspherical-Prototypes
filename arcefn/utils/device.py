"""
Cross-Platform Device Detection
===============================
Auto-detects available compute backend in priority order:
  1. DirectML  (AMD GPU on Windows)   — via torch_directml
  2. CUDA      (NVIDIA GPU)            — via torch.cuda
  3. CPU       (fallback)

All training/evaluation scripts import from here so there is a single
point of change for device management across the entire project.

Usage:
    from arcefn.utils.device import get_device, empty_cache, amp_available, get_amp_scaler
    device, backend = get_device()

NOTE (this machine): the project venv pins ``torch==2.3.1`` +
``torch-directml`` (AMD Radeon RX 5700 XT).  torch-directml is
version-locked to the torch 2.3.x series: upgrading torch beyond 2.3.x
breaks the DML plugin and silently falls back to CPU.  Also,
``torch.__version__`` reporting ``2.3.1+cpu`` is expected and correct
for DirectML installs (the plugin sits on top of the CPU wheel) — do
not "fix" it by reinstalling a different torch build.
"""
import torch


def get_device() -> tuple[torch.device, str]:
    """
    Detect and return the best available compute device.

    Returns
    -------
    device : torch.device
        The PyTorch device object (DirectML, CUDA, or CPU).
    backend_type : str
        One of ``'dml'``, ``'cuda'``, or ``'cpu'`` — used to gate
        backend-specific operations (AMP, cache clearing, etc.).

    Env override: ``ARCEFN_DEVICE=cuda|dml|cpu`` forces a backend (case-insensitive).
    """
    import os as _os
    _forced = _os.environ.get("ARCEFN_DEVICE", "").strip().lower()
    if _forced in ("cuda", "dml", "cpu"):
        if _forced == "cuda" and torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if _forced == "dml":
            try:
                import torch_directml  # noqa: F401
                return torch_directml.device(), "dml"
            except Exception:
                pass
        if _forced == "cpu":
            return torch.device("cpu"), "cpu"
    # Priority 1: DirectML (AMD GPU on Windows)
    # Requires: pip install torch-directml — and numpy<2, because the DML
    # plugin is compiled against NumPy 1.x and its import fails under
    # NumPy 2.x (RuntimeError, not ImportError → silent CPU fallback).
    try:
        import torch_directml  # noqa: F401
        dml_device = torch_directml.device()
        print(f"  Using DirectML device (AMD GPU): {dml_device}")
        return dml_device, 'dml'
    except ImportError:
        pass
    except Exception as exc:
        print(
            f"  WARNING: torch-directml import failed ({exc!r}) — falling back to CUDA/CPU. "
            "If this machine has an AMD GPU, downgrade numpy: pip install 'numpy<2'"
        )

    # Priority 2: CUDA (NVIDIA GPU on any OS)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device, 'cuda'

    # Priority 3: CPU (fallback)
    print("  Using CPU (no GPU detected)")
    return torch.device('cpu'), 'cpu'


def empty_cache(backend_type: str) -> None:
    """
    Clear GPU memory cache for the active backend.

    This is a no-op for backends that do not expose an explicit cache
    management API (DirectML, CPU).

    Parameters
    ----------
    backend_type : str
        One of ``'dml'``, ``'cuda'``, or ``'cpu'``.
    """
    if backend_type == 'cuda':
        torch.cuda.empty_cache()
    # DirectML: no explicit cache-clearing call needed.
    # CPU: no cache to clear.


def amp_available(backend_type: str) -> bool:
    """
    Check whether automatic mixed precision (AMP) is available.

    Only CUDA provides ``torch.cuda.amp``.  DirectML and CPU fall back
    to full FP32.

    Parameters
    ----------
    backend_type : str
        One of ``'dml'``, ``'cuda'``, or ``'cpu'``.

    Returns
    -------
    bool
    """
    return backend_type == 'cuda'


def get_amp_scaler(backend_type: str, enabled: bool = True) -> torch.cuda.amp.GradScaler | None:
    """
    Return a ``torch.cuda.amp.GradScaler`` when AMP is both available
    and requested; return ``None`` otherwise.

    Parameters
    ----------
    backend_type : str
        One of ``'dml'``, ``'cuda'``, or ``'cpu'``.
    enabled : bool
        Whether the caller wants to use AMP (subject to backend support).

    Returns
    -------
    torch.cuda.amp.GradScaler or None
    """
    if enabled and backend_type == 'cuda':
        return torch.cuda.amp.GradScaler()
    return None
