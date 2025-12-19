"""witnessed_ph.schema

Lightweight data-model definitions used across the package.

We keep these as plain TypedDicts / dicts so that:
- results are JSON-serialisable with minimal fuss
- the library stays friendly to notebooks/scripts

The book's Chapter 4 refers to:
- token occurrences (TokenID)
- measurement locations (utterances / turns)
- witnessed bars (k, b, d, ρ) where ρ contains witnesses and (optionally) a representative cycle.

This module names those concepts in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Basic objects: tokens and utterances
# ---------------------------------------------------------------------------

class Token(TypedDict):
    id: str
    text: str
    lemma: str
    pos: str
    utterance_id: str
    char_start: int
    char_end: int
    embedding: NDArray[np.float32]


class Utterance(TypedDict):
    id: str
    speaker: Optional[str]
    text: str
    token_ids: List[str]
    embedding: NDArray[np.float32]


class PointCloudData(TypedDict):
    """Output of `embedding.text_to_point_cloud`."""
    embeddings: NDArray[np.float32]          # (N, d) unit-normalised
    token_ids: List[str]                     # index -> token_id
    tokens: Dict[str, Token]                 # token_id -> Token
    utterances: Dict[str, Utterance]         # utterance_id -> Utterance


# ---------------------------------------------------------------------------
# Witnessed bars and diagrams
# ---------------------------------------------------------------------------

class Cycle1(TypedDict, total=False):
    """A representative 1-cycle, stored as edges between vertex indices."""
    edges: List[Tuple[int, int]]
    simplices: List[Tuple[int, ...]]  # optional richer form


class Witness(TypedDict, total=False):
    """Witness payload for a bar."""
    token_ids: List[str]                 # token occurrence IDs
    tokens: Dict[str, List[Any]]         # convenient split into surface / lemma / pos etc
    utterance_ids: List[str]             # measurement locations touched by the witness
    centroid: List[float]                # embedding centroid (as python list for JSON)
    cycle: Cycle1                        # for H1 (optional)


class Bar(TypedDict, total=False):
    id: str
    dim: int
    birth: float
    death: float
    persistence: float
    witness: Witness


class Diagram(TypedDict, total=False):
    """A witnessed persistence diagram for one slice."""
    slice_id: str
    num_tokens: int
    num_utterances: int
    config: Dict[str, Any]
    bars: List[Bar]
    # diagnostics / provenance
    library_versions: Dict[str, str]
    embedding_model: str
    distance_metric: str
    notes: str


Config = Dict[str, Any]


# Backwards-compat: older code imported default_config from schema
from .config import default_config  # noqa: E402,F401
