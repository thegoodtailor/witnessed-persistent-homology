"""witnessed_ph.h0

0-dimensional persistence and *theme bars*.

Why this exists:

In standard Vietoris–Rips persistent homology, every point creates an H0 class
born at r=0. Those singleton bars are mathematically correct but semantically
uninteresting for text analysis.

Chapter 4's *witnessed theme* view focuses instead on multi-token clusters:
components that become meaningful only once at least `min_witness_tokens`
tokens have merged. In this view, an H0 bar can have birth > 0: it is the first
scale at which that cluster exists as a non-trivial, witnessable theme.

Implementation strategy:

We compute the single-linkage merge tree induced by the distance matrix
(Kruskal / union-find). Internal nodes correspond to clusters formed at some
merge radius. Each internal node yields a candidate theme bar:

    birth = merge radius when the cluster first forms
    death = merge radius when the cluster merges into a larger one
    witnesses = tokens contained in the cluster (optionally truncated for display)

This is a deterministic, lightweight, and reproducible construction that stays
faithful to the interpretive intent of the chapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import angular_distance


@dataclass
class _ClusterNode:
    id: int
    birth: float
    death: Optional[float]
    members: List[int]               # vertex indices (0..N-1)
    children: Optional[Tuple[int, int]] = None


def _union_find_make(n: int) -> Dict[int, int]:
    return {i: i for i in range(n)}


def _uf_find(parent: Dict[int, int], x: int) -> int:
    # path compression
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def compute_h0_theme_bars(
    embeddings: NDArray[np.float32],
    token_ids: List[str],
    tokens: Dict[str, Any],
    distance_matrix: NDArray[np.float32],
    *,
    max_edge_length: float,
    min_persistence: float,
    min_witness_tokens: int,
    max_witness_tokens: Optional[int] = 5,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    """Compute witnessed H0 *theme bars* from a distance matrix.

    Parameters
    ----------
    embeddings:
        (N, d) unit-normalised token embeddings.
    token_ids:
        index -> token_id
    tokens:
        token_id -> Token dict (must include text/lemma/pos/utterance_id)
    distance_matrix:
        (N, N) symmetric, 0 diagonal.
    max_edge_length:
        Ignore edges beyond this length (filtration truncation).
    min_persistence:
        Filter bars with death - birth < min_persistence.
    min_witness_tokens:
        Filter clusters with fewer than this many tokens.
    max_witness_tokens:
        Truncate displayed witness list to this many tokens (None to keep all).
    distance_metric:
        "cosine" uses angular distance for centroid selection; "euclidean" uses L2.

    Returns
    -------
    List of bar dicts (dim=0).
    """
    n = int(distance_matrix.shape[0])
    if n == 0:
        return []

    # Build edge list
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance_matrix[i, j])
            if d <= max_edge_length:
                edges.append((d, i, j))
    edges.sort(key=lambda t: t[0])

    # Initialise singleton cluster nodes
    nodes: Dict[int, _ClusterNode] = {}
    parent: Dict[int, int] = _union_find_make(n)
    next_id = n

    for i in range(n):
        nodes[i] = _ClusterNode(id=i, birth=0.0, death=None, members=[i], children=None)

    # Kruskal merges
    for w, i, j in edges:
        ri, rj = _uf_find(parent, i), _uf_find(parent, j)
        if ri == rj:
            continue

        # create new node
        new_members = nodes[ri].members + nodes[rj].members
        new_node = _ClusterNode(id=next_id, birth=float(w), death=None, members=new_members, children=(ri, rj))
        nodes[next_id] = new_node

        # children die now
        if nodes[ri].death is None:
            nodes[ri].death = float(w)
        if nodes[rj].death is None:
            nodes[rj].death = float(w)

        # union: parent of roots -> new node
        parent[ri] = next_id
        parent[rj] = next_id
        parent[next_id] = next_id

        next_id += 1

    # Any remaining roots die at max_edge_length (filtration truncation)
    for node in nodes.values():
        if node.death is None:
            node.death = float(max_edge_length)

    # Turn cluster nodes into bars
    bars: List[Dict[str, Any]] = []
    bar_idx = 0

    for node_id, node in nodes.items():
        if len(node.members) < int(min_witness_tokens):
            continue
        birth = float(node.birth)
        death = float(node.death if node.death is not None else max_edge_length)
        persistence = death - birth
        if persistence < float(min_persistence):
            continue

        member_token_ids_full = [token_ids[m] for m in node.members]

        # Compute centroid over all members
        member_vecs = embeddings[node.members]
        centroid = member_vecs.mean(axis=0)
        # normalise centroid (avoid divide-by-zero)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        # Pick a readable subset of witness tokens
        if max_witness_tokens is None or len(node.members) <= max_witness_tokens:
            chosen_members = node.members
        else:
            # distance to centroid, choose closest
            if distance_metric == "euclidean":
                dists = np.linalg.norm(member_vecs - centroid[None, :], axis=1)
            else:
                # angular distance is stable for unit vectors
                dists = np.array([angular_distance(member_vecs[k], centroid) for k in range(member_vecs.shape[0])], dtype=np.float32)
            order = np.argsort(dists)
            chosen_members = [node.members[int(k)] for k in order[: int(max_witness_tokens)]]

        chosen_token_ids = [token_ids[m] for m in chosen_members]

        # Build witness payload
        witness_tokens = [tokens[tok_id] for tok_id in chosen_token_ids]
        member_tokens_full = [tokens[tok_id] for tok_id in member_token_ids_full]
        lemma_full = [t.get("lemma") for t in member_tokens_full]
        surface_full = [t.get("text") for t in member_tokens_full]
        witness = {
            "token_ids": chosen_token_ids,
            "token_ids_full": member_token_ids_full,
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
        }

        bars.append(
            {
                "id": f"b{bar_idx}",
                "dim": 0,
                "birth": birth,
                "death": death,
                "persistence": persistence,
                "witness": witness,
            }
        )
        bar_idx += 1

    # Sort bars by persistence descending (book tends to show long bars)
    bars.sort(key=lambda b: float(b.get("persistence", 0.0)), reverse=True)
    return bars


def compute_h0_raw_bars(
    embeddings: NDArray[np.float32],
    token_ids: List[str],
    tokens: Dict[str, Any],
    distance_matrix: NDArray[np.float32],
    *,
    max_edge_length: float,
    min_persistence: float,
    min_witness_tokens: int,
    max_witness_tokens: Optional[int] = 5,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    """Compute *standard* H0 persistence bars (all births are 0).

    This is mainly for debugging / sanity-checking the distinction between:
    - the raw H0 diagram D_0 (singleton births at 0), and
    - the Chapter-4 theme view (compute_h0_theme_bars) where births can be >0.

    The implementation uses a Kruskal-style union-find with a deterministic
    elder rule: the component whose representative has the smaller vertex index
    survives each merge.

    Note:
    - If the truncated filtration (max_edge_length) does not connect the graph,
      multiple "infinite" bars are returned with death=max_edge_length.
    - Witness sets use the token set of the dying component at merge time.
    """
    n = int(distance_matrix.shape[0])
    if n == 0:
        return []

    # Build edge list
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance_matrix[i, j])
            if d <= max_edge_length:
                edges.append((d, i, j))
    edges.sort(key=lambda t: t[0])

    # Union-find over vertices
    parent = {i: i for i in range(n)}
    # Component members for witness sets
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    # Representative vertex index (elder rule): min index in component
    rep: Dict[int, int] = {i: i for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bars: List[Dict[str, Any]] = []
    bar_idx = 0

    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue

        # Determine survivor / dying component deterministically
        if rep[ri] <= rep[rj]:
            survivor, dying = ri, rj
        else:
            survivor, dying = rj, ri

        # Dying component's bar ends now
        dying_members = members[dying]
        if len(dying_members) >= int(min_witness_tokens):
            birth = 0.0
            death = float(w)
            persistence = death - birth
            if persistence >= float(min_persistence):
                member_token_ids_full = [token_ids[m] for m in dying_members]
                member_tokens_full = [tokens[tok_id] for tok_id in member_token_ids_full]
                lemma_full = [t.get("lemma") for t in member_tokens_full]
                surface_full = [t.get("text") for t in member_tokens_full]

                # centroid over all members
                vecs = embeddings[dying_members]
                centroid = vecs.mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm > 0:
                    centroid = centroid / norm

                # Choose readable witnesses
                if max_witness_tokens is None or len(dying_members) <= max_witness_tokens:
                    chosen_members = dying_members
                else:
                    if distance_metric == "euclidean":
                        dists = np.linalg.norm(vecs - centroid[None, :], axis=1)
                    else:
                        dists = np.array([angular_distance(vecs[k], centroid) for k in range(vecs.shape[0])], dtype=np.float32)
                    order = np.argsort(dists)
                    chosen_members = [dying_members[int(k)] for k in order[: int(max_witness_tokens)]]

                chosen_token_ids = [token_ids[m] for m in chosen_members]
                witness_tokens = [tokens[tok_id] for tok_id in chosen_token_ids]

                witness = {
                    "token_ids": chosen_token_ids,
                    "token_ids_full": member_token_ids_full,
                    "tokens_full": {"surface": surface_full, "lemma": lemma_full},
                    "lemma_set": sorted({(l or "").lower() for l in lemma_full if l}),
                    "tokens": {
                        "surface": [t.get("text") for t in witness_tokens],
                        "lemma": [t.get("lemma") for t in witness_tokens],
                        "pos": [t.get("pos") for t in witness_tokens],
                    },
                    "utterance_ids": sorted({t.get("utterance_id") for t in witness_tokens if t.get("utterance_id") is not None}),
                    "centroid": centroid.astype(np.float32).tolist(),
                }

                bars.append(
                    {
                        "id": f"b{bar_idx}",
                        "dim": 0,
                        "birth": birth,
                        "death": death,
                        "persistence": persistence,
                        "witness": witness,
                    }
                )
                bar_idx += 1

        # Union components
        parent[dying] = survivor
        members[survivor] = members[survivor] + members[dying]
        rep[survivor] = min(rep[survivor], rep[dying])

    # Remaining components: truncated "infinite" bars
    roots = sorted({find(i) for i in range(n)})
    for r in roots:
        comp_members = members[r]
        if len(comp_members) < int(min_witness_tokens):
            continue
        birth = 0.0
        death = float(max_edge_length)
        persistence = death - birth
        if persistence < float(min_persistence):
            continue

        member_token_ids_full = [token_ids[m] for m in comp_members]
        member_tokens_full = [tokens[tok_id] for tok_id in member_token_ids_full]
        lemma_full = [t.get("lemma") for t in member_tokens_full]
        surface_full = [t.get("text") for t in member_tokens_full]

        vecs = embeddings[comp_members]
        centroid = vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        if max_witness_tokens is None or len(comp_members) <= max_witness_tokens:
            chosen_members = comp_members
        else:
            if distance_metric == "euclidean":
                dists = np.linalg.norm(vecs - centroid[None, :], axis=1)
            else:
                dists = np.array([angular_distance(vecs[k], centroid) for k in range(vecs.shape[0])], dtype=np.float32)
            order = np.argsort(dists)
            chosen_members = [comp_members[int(k)] for k in order[: int(max_witness_tokens)]]

        chosen_token_ids = [token_ids[m] for m in chosen_members]
        witness_tokens = [tokens[tok_id] for tok_id in chosen_token_ids]

        witness = {
            "token_ids": chosen_token_ids,
            "token_ids_full": member_token_ids_full,
            "tokens_full": {"surface": surface_full, "lemma": lemma_full},
            "lemma_set": sorted({(l or "").lower() for l in lemma_full if l}),
            "tokens": {
                "surface": [t.get("text") for t in witness_tokens],
                "lemma": [t.get("lemma") for t in witness_tokens],
                "pos": [t.get("pos") for t in witness_tokens],
            },
            "utterance_ids": sorted({t.get("utterance_id") for t in witness_tokens if t.get("utterance_id") is not None}),
            "centroid": centroid.astype(np.float32).tolist(),
        }

        bars.append(
            {
                "id": f"b{bar_idx}",
                "dim": 0,
                "birth": birth,
                "death": death,
                "persistence": persistence,
                "witness": witness,
            }
        )
        bar_idx += 1

    bars.sort(key=lambda b: float(b.get("persistence", 0.0)), reverse=True)
    return bars
