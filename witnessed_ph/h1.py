"""witnessed_ph.h1

Optional 1-dimensional persistence (loops) via GUDHI.

Chapter 4 treats H1 as "toy-interesting but not load-bearing" for the temporal
calculus, but H1 can still produce striking, interpretable loops in small texts.

We keep this module intentionally conservative:
- if `gudhi` is not installed, we skip H1 gracefully
- witness extraction is deterministic but intentionally simple:
  we use the vertices of the *death simplex* (often a triangle) as the witness set,
  and (when that simplex is a triangle) we use its boundary edges as a concrete
  representative cycle.

This is not the only possible witness-extraction policy, but it is stable and
easy to reproduce.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import angular_distance


def _try_import_gudhi():
    try:
        import gudhi  # type: ignore
        return gudhi
    except Exception:
        return None


def compute_h1_bars_gudhi(
    embeddings: NDArray[np.float32],
    token_ids: List[str],
    tokens: Dict[str, Any],
    distance_matrix: NDArray[np.float32],
    *,
    max_edge_length: float,
    min_persistence: float,
    max_witness_tokens: Optional[int] = 5,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    """Compute witnessed H1 bars using GUDHI's Vietoris–Rips complex.

    Returns an empty list if gudhi is not installed.
    """
    gudhi = _try_import_gudhi()
    if gudhi is None:
        return []

    # Need simplices up to dimension 2 to kill 1-cycles.
    rips = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=2)

    st.persistence()

    bars: List[Dict[str, Any]] = []
    idx = 0

    try:
        pairs = st.persistence_pairs()
    except Exception:
        # Older gudhi versions may not expose persistence_pairs
        pairs = []

    for birth_sx, death_sx in pairs:
        dim = len(birth_sx) - 1
        if dim != 1:
            continue

        birth = float(st.filtration(birth_sx))
        if death_sx is None or len(death_sx) == 0:
            death = float(max_edge_length)
        else:
            death = float(st.filtration(death_sx))

        persistence = death - birth
        if persistence < float(min_persistence):
            continue

        # Witness vertices: use death simplex vertices when available, else birth simplex.
        if death_sx is None or len(death_sx) == 0:
            verts = list(birth_sx)
        else:
            verts = list(death_sx)

        # Compute centroid for semantic comparisons
        vecs = embeddings[verts]
        centroid = vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        # Choose readable witnesses
        if max_witness_tokens is None or len(verts) <= max_witness_tokens:
            chosen = verts
        else:
            if distance_metric == "euclidean":
                dists = np.linalg.norm(vecs - centroid[None, :], axis=1)
            else:
                dists = np.array([angular_distance(vecs[k], centroid) for k in range(vecs.shape[0])], dtype=np.float32)
            order = np.argsort(dists)
            chosen = [verts[int(k)] for k in order[: int(max_witness_tokens)]]

        chosen_token_ids = [token_ids[v] for v in chosen]
        witness_tokens = [tokens[tok_id] for tok_id in chosen_token_ids]
        full_token_ids = [token_ids[v] for v in verts]
        member_tokens_full = [tokens[tok_id] for tok_id in full_token_ids]
        lemma_full = [t.get("lemma") for t in member_tokens_full]
        surface_full = [t.get("text") for t in member_tokens_full]

        # A concrete cycle when the death simplex is a triangle
        cycle_edges: List[Tuple[int, int]] = []
        if death_sx is not None and len(death_sx) == 3:
            a, b, c = death_sx
            cycle_edges = [(int(a), int(b)), (int(b), int(c)), (int(a), int(c))]

        witness = {
            "token_ids": chosen_token_ids,
            "token_ids_full": full_token_ids,
            "tokens_full": {
                "surface": surface_full,
                "lemma": lemma_full,
            },
            "lemma_set": sorted({(l or "").lower() for l in lemma_full if l}),
            "tokens": {
                "surface": [t.get("text") for t in witness_tokens],
                "lemma": [t.get("lemma") for t in witness_tokens],
                "pos": [t.get("pos") for t in witness_tokens],
            },
            "utterance_ids": sorted({t.get("utterance_id") for t in witness_tokens if t.get("utterance_id") is not None}),
            "centroid": centroid.astype(np.float32).tolist(),
            "cycle": {"edges": cycle_edges},
        }

        bars.append(
            {
                "id": f"b{idx}",
                "dim": 1,
                "birth": birth,
                "death": death,
                "persistence": persistence,
                "witness": witness,
            }
        )
        idx += 1

    bars.sort(key=lambda b: float(b.get("persistence", 0.0)), reverse=True)
    return bars
