"""witnessed_ph.utils

Small utilities used throughout the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import math

import numpy as np
from numpy.typing import NDArray


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_distance(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """1 - cosine similarity, in [0,2]."""
    # ensure float
    u = u.astype(np.float32, copy=False)
    v = v.astype(np.float32, copy=False)
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 1.0
    sim = float(np.dot(u, v) / denom)
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def angular_distance(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """arccos(cos sim)/pi in [0,1] for unit vectors."""
    u = u.astype(np.float32, copy=False)
    v = v.astype(np.float32, copy=False)
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.5
    sim = float(np.dot(u, v) / denom)
    sim = max(-1.0, min(1.0, sim))
    return float(math.acos(sim) / math.pi)


def ensure_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types into JSON-friendly python types."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    # fall back to string
    return str(obj)
