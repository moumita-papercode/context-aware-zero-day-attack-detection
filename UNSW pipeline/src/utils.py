from __future__ import annotations

import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

START_TS = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_environment(seed_np: int = 42, seed_torch: int = 42, seed_cuda: int = 44) -> None:
    np.random.seed(seed_np)
    torch.manual_seed(seed_torch)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed_cuda)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            track(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            track("CUDA available, but GPU name could not be read")
    else:
        track("Using CPU")


def track(msg: str, start_ts: Optional[float] = None) -> None:
    ts = START_TS if start_ts is None else start_ts
    elapsed = time.time() - ts
    print(f"[{time.strftime('%H:%M:%S')}] (+{elapsed:,.1f}s) {msg}")
