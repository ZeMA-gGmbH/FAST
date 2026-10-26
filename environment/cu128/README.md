# CUDA 12.8 training environment

Blackwell-compatible (sm_120) runtime for the D0/D1/D2 retraining of the
component ablation (retraining without action features). Every shared pin
equals `environment/cu121`; only Python (3.10.20), torch (2.7.1+cu128 from the
official PyTorch cu128 index), torch-geometric (2.7.0, the first release that
officially supports torch 2.7), and the optional W&B telemetry mirror (0.30.0)
differ. Python 3.8 is impossible here because torch 2.7 requires Python >= 3.9,
and PyG 2.7 requires Python >= 3.10. Resolution uses `exclude-newer` 2026-09-25.

```bash
UV_PROJECT_ENVIRONMENT=.venv-cu128 \
  uv sync --project environment/cu128 --frozen --no-install-project --python 3.10.20
```

Set `FAST_COMPAT_PYTHON` to the Python of the `environment/cu121` runtime; the
runner records its versions. W&B is never required: `--telemetry disabled` is
the default, and `offline` keeps the mirror local.

Under WSL2 the GPU path uses the Windows host driver (`/usr/lib/wsl/lib/libcuda.so`);
no Linux NVIDIA driver or CUDA toolkit is installed inside WSL.
