from __future__ import annotations

import random
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

START_TS = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def track(msg: str) -> None:
    elapsed = time.time() - START_TS
    print(f"[{time.strftime('%H:%M:%S')}] (+{elapsed:,.1f}s) {msg}")


def setup_environment(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(44)
        try:
            gpu_name = torch.cuda.get_device_name(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            track(f"Using GPU: {gpu_name}")
        except Exception:
            track("CUDA unavailable")
    else:
        track("Using CPU")
