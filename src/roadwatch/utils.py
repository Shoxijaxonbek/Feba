"""Determinism and time-budget helpers."""
from __future__ import annotations

import os
import random
import time
from collections import deque

import numpy as np


class Pace:
    """Projects the finishing time of a long loop from its *recent* rate.

    Startup costs (model warm-up, first decode) are excluded by a warm-up period
    and by using only the last `window` progress marks, so the projection reflects
    the steady state. Call `reset_window()` after changing the workload (e.g.
    thinning), so the next decision is based on the new rate only.
    """

    def __init__(self, budget_s: float | None, total: float, warmup: float, window: int = 8):
        self.budget_s, self.total, self.warmup = budget_s, total, warmup
        self.t0 = time.perf_counter()
        self.marks: deque = deque(maxlen=window)

    def reset_window(self) -> None:
        self.marks.clear()

    def over_budget(self, done: float) -> bool:
        """Record progress (`done` units of `total`); True if the projected end exceeds the budget."""
        now = time.perf_counter() - self.t0
        self.marks.append((done, now))
        if not self.budget_s or done < self.warmup or len(self.marks) < self.marks.maxlen:
            return False
        d0, t0 = self.marks[0]
        rate = (now - t0) / max(done - d0, 1e-9)
        return now + (self.total - done) * rate > self.budget_s


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
