"""Determinism helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Fix every RNG we touch. Inference is otherwise deterministic up to cuDNN float noise."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
