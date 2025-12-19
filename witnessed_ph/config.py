"""witnessed_ph.config

Centralised configuration + reproducibility helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import random

import numpy as np


def default_config() -> Dict[str, Any]:
    """Return a *copy* of the default configuration.

    The defaults aim to match the narrative choices in Chapter 4:
    - contextual embeddings from DeBERTa-v3-base (HuggingFace)
    - content-token filtering (NOUN/VERB/ADJ/PROPN)
    - angular/cosine-based distance on unit-normalised embeddings
    - focus on H0 themes, with optional H1 loops

    You can override any key in the returned dict.
    """
    return {
        # --- tokenisation / preprocessing ---
        "pos_filter": ["NOUN", "VERB", "ADJ", "PROPN"],
        "min_token_len": 3,
        "use_lemmas": True,
        "stopwords": [],  # optionally extend
        "strip_speaker_labels": True,  # remove "User:" / "Assistant:" prefix if present

        # --- embedding ---
        "embedding_model": "microsoft/deberta-v3-base",
        "spacy_model": "en_core_web_sm",
        "device": "auto",  # "auto" | "cpu" | "cuda"
        "max_length": 512,

        # --- distance / filtration ---
        "distance_metric": "cosine",  # "cosine" (angular) | "euclidean"
        "max_edge_length": 0.75,        # filtration truncation (angular distance is in [0,1])
        "max_dimension": 1,             # compute H0 and H1

        # --- H0 mode ---
        # "theme" gives Chapter-4 style multi-token theme bars (births may be >0)
        # "raw" gives standard H0 persistence bars (all births = 0)
        "h0_mode": "theme",

        # --- witnessed bar selection ---
        # Interpretable diagrams focus on bars that are (a) persistent and (b) have non-trivial witnesses.
        "min_persistence": 0.03,
        "min_witness_tokens": 2,
        "max_witness_tokens": 5,        # truncate witness token lists for readability (None for no truncation)

        # --- temporal tracking (Chapter 4.8 style) ---
        "lambda_sem": 0.5,              # λ in d_bar = max(||Δ endpoints||_∞, λ·d_sem)
        "epsilon_match": 0.8,           # admissible matching threshold
        "theta_carry": 0.4,             # Jaccard ≥ θ => carry; else drift (if admissible)
        "delta_sem_max": 0.6,           # semantic drift bound used as a sanity-check
        "topo_endpoint_eps": 0.2,       # allowable endpoint movement (b and d separately)

        # --- reproducibility ---
        "random_seed": 0,
        "torch_deterministic": True,
    }


def set_global_seeds(seed: int) -> None:
    """Best-effort reproducibility across numpy / python / torch."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Optional determinism flags
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def get_library_versions() -> Dict[str, str]:
    """Collect versions of key libraries for provenance."""
    versions: Dict[str, str] = {}

    def _add(pkg: str) -> None:
        try:
            import importlib.metadata as md  # py>=3.8
            versions[pkg] = md.version(pkg)
        except Exception:
            pass

    for pkg in ["numpy", "scipy", "sklearn", "torch", "transformers", "spacy", "gudhi"]:
        _add(pkg)
    return versions


def resolve_device(device: str) -> str:
    """Resolve "auto" into "cpu"/"cuda" if torch is installed."""
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
