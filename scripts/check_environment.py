from __future__ import annotations
import sys
try:
    import torch
    import torchvision
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    else:
        print("WARNING: CUDA не обнаружена; обучение будет на CPU и займёт существенно больше времени.")
except Exception as exc:
    raise SystemExit(f"Ошибка окружения: {type(exc).__name__}: {exc}")
