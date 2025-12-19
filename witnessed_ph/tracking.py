"""witnessed_ph.tracking

A lightweight implementation of the Chapter 4 *bar dynamics* layer.

The book discusses carry / drift / rupture / re-entry events between slices.
This module provides a pragmatic implementation that is:

- deterministic
- explainable (all matches are explicit)
- good enough to reproduce the *type* of experiment in Chapter 4.8

Important note:
This is a model of the calculus, not a claim that this is the only correct one.
The key is to be explicit about the matching metric and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import angular_distance, jaccard


def _bar_centroid(bar: Dict[str, Any]) -> NDArray[np.float32]:
    w = bar.get("witness", {}) or {}
    c = w.get("centroid", None)
    if c is None:
        return np.zeros((1,), dtype=np.float32)
    arr = np.array(c, dtype=np.float32)
    return arr


def _bar_lemma_set(bar: Dict[str, Any]) -> List[str]:
    w = bar.get("witness", {}) or {}
    lemmas = w.get("lemma_set", None)
    if lemmas is None:
        # fall back to displayed witness lemmas
        lemmas = (w.get("tokens", {}) or {}).get("lemma", []) or []
    # normalise
    return [str(x).lower() for x in lemmas if x is not None and str(x).strip()]


def bar_distance(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    lambda_sem: float,
) -> Tuple[float, float, float]:
    """Return (d_bar, topo_dist, sem_dist)."""
    a_birth, a_death = float(a.get("birth", 0.0)), float(a.get("death", 0.0))
    b_birth, b_death = float(b.get("birth", 0.0)), float(b.get("death", 0.0))

    topo = max(abs(a_birth - b_birth), abs(a_death - b_death))

    ca, cb = _bar_centroid(a), _bar_centroid(b)
    if ca.shape != cb.shape or ca.size == 0:
        sem = 0.0
    else:
        sem = float(angular_distance(ca, cb))

    dbar = max(topo, float(lambda_sem) * sem)
    return dbar, topo, sem


@dataclass
class Journey:
    id: int
    dim: int
    anchor_slice: int
    anchor_bar: Dict[str, Any]
    status: str  # "active" | "ruptured"
    current_bar: Optional[Dict[str, Any]]
    events: List[Dict[str, Any]]


def _best_match(
    source_bar: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    lambda_sem: float,
    epsilon_match: float,
    topo_endpoint_eps: float,
) -> Optional[Tuple[int, Dict[str, Any], float, float, float]]:
    best = None
    for idx, cand in enumerate(candidates):
        dbar, topo, sem = bar_distance(source_bar, cand, lambda_sem=lambda_sem)
        if dbar > epsilon_match:
            continue
        # optional stricter endpoint control
        if abs(float(source_bar.get("birth", 0.0)) - float(cand.get("birth", 0.0))) > topo_endpoint_eps:
            continue
        if abs(float(source_bar.get("death", 0.0)) - float(cand.get("death", 0.0))) > topo_endpoint_eps:
            continue
        if best is None or dbar < best[2]:
            best = (idx, cand, dbar, topo, sem)
    return best


def track_bars_over_time(
    diagrams: Sequence[Dict[str, Any]],
    *,
    dim: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Track bars across a sequence of slice diagrams.

    Parameters
    ----------
    diagrams:
        Output of `analyse_text_single_slice` for successive slices τ0, τ1, ...
    dim:
        Which homology dimension to track (Chapter 4 experiments mainly track H0).
    config:
        Thresholds; if None uses the config embedded in the first diagram.

    Returns
    -------
    dict with:
        journeys: list of journeys
        transitions: per-step event counts and matched IDs
    """
    if not diagrams:
        return {"journeys": [], "transitions": []}

    cfg = dict(diagrams[0].get("config", {}))
    if config:
        cfg.update(config)

    lambda_sem = float(cfg.get("lambda_sem", 0.5))
    epsilon_match = float(cfg.get("epsilon_match", 0.8))
    theta_carry = float(cfg.get("theta_carry", 0.4))
    delta_sem_max = float(cfg.get("delta_sem_max", 0.6))
    topo_endpoint_eps = float(cfg.get("topo_endpoint_eps", 0.2))

    # Helper: extract bars of the chosen dimension
    def bars_in(diag: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [b for b in diag.get("bars", []) if int(b.get("dim", -1)) == int(dim)]

    # Initialise journeys at τ0
    journeys: List[Journey] = []
    next_jid = 0
    slice0 = bars_in(diagrams[0])
    for b in slice0:
        journeys.append(
            Journey(
                id=next_jid,
                dim=int(dim),
                anchor_slice=0,
                anchor_bar=b,
                status="active",
                current_bar=b,
                events=[{"t": 0, "type": "birth", "bar_id": b.get("id")}],
            )
        )
        next_jid += 1

    transitions: List[Dict[str, Any]] = []

    # Iterate transitions τ_t -> τ_{t+1}
    for t in range(len(diagrams) - 1):
        prev_diag = diagrams[t]
        cur_diag = diagrams[t + 1]
        prev_bars = bars_in(prev_diag)
        cur_bars_all = bars_in(cur_diag)

        # Keep track of which current bars are already claimed
        claimed: set[str] = set()

        step = {
            "from": prev_diag.get("slice_id", f"τ{t}"),
            "to": cur_diag.get("slice_id", f"τ{t+1}"),
            "carry": 0,
            "drift": 0,
            "rupture": 0,
            "reentry": 0,
            "birth": 0,
            "matches": [],  # list of (journey_id, prev_bar_id, cur_bar_id, event_type, dbar, topo, sem, jaccard)
        }

        # ---- 1) Continue active journeys (match from previous current_bar) ----
        for j in journeys:
            if j.dim != int(dim) or j.status != "active" or j.current_bar is None:
                continue

            # candidates not yet claimed
            candidates = [b for b in cur_bars_all if str(b.get("id")) not in claimed]
            best = _best_match(
                j.current_bar,
                candidates,
                lambda_sem=lambda_sem,
                epsilon_match=epsilon_match,
                topo_endpoint_eps=topo_endpoint_eps,
            )

            if best is None:
                j.status = "ruptured"
                j.current_bar = None
                j.events.append({"t": t + 1, "type": "rupture", "from": prev_diag.get("slice_id")})
                step["rupture"] += 1
                continue

            _, chosen, dbar, topo, sem = best
            # drift sanity-check
            if sem > delta_sem_max:
                j.status = "ruptured"
                j.current_bar = None
                j.events.append({"t": t + 1, "type": "rupture_semantic", "sem": sem})
                step["rupture"] += 1
                continue

            # classify carry vs drift via Jaccard overlap of lemma sets
            jac = jaccard(_bar_lemma_set(j.current_bar), _bar_lemma_set(chosen))
            if jac >= theta_carry:
                etype = "carry"
                step["carry"] += 1
            else:
                etype = "drift"
                step["drift"] += 1

            claimed.add(str(chosen.get("id")))
            step["matches"].append((j.id, j.current_bar.get("id"), chosen.get("id"), etype, dbar, topo, sem, jac))
            j.current_bar = chosen
            j.events.append({"t": t + 1, "type": etype, "bar_id": chosen.get("id"), "dbar": dbar, "jaccard": jac})

        # ---- 2) Re-entry for ruptured journeys (match from anchor_bar) ----
        for j in journeys:
            if j.dim != int(dim) or j.status != "ruptured":
                continue

            candidates = [b for b in cur_bars_all if str(b.get("id")) not in claimed]
            best = _best_match(
                j.anchor_bar,
                candidates,
                lambda_sem=lambda_sem,
                epsilon_match=epsilon_match,
                topo_endpoint_eps=topo_endpoint_eps,
            )
            if best is None:
                continue

            _, chosen, dbar, topo, sem = best
            if sem > delta_sem_max:
                continue

            claimed.add(str(chosen.get("id")))
            step["reentry"] += 1
            step["matches"].append((j.id, None, chosen.get("id"), "reentry", dbar, topo, sem, None))
            j.status = "active"
            j.current_bar = chosen
            j.events.append({"t": t + 1, "type": "reentry", "bar_id": chosen.get("id"), "dbar": dbar})

        # ---- 3) Any unclaimed current bars are births ----
        for b in cur_bars_all:
            if str(b.get("id")) in claimed:
                continue
            journeys.append(
                Journey(
                    id=next_jid,
                    dim=int(dim),
                    anchor_slice=t + 1,
                    anchor_bar=b,
                    status="active",
                    current_bar=b,
                    events=[{"t": t + 1, "type": "birth", "bar_id": b.get("id")}],
                )
            )
            next_jid += 1
            step["birth"] += 1

        transitions.append(step)

    # Convert dataclasses to JSONable dicts
    journeys_out: List[Dict[str, Any]] = []
    for j in journeys:
        journeys_out.append(
            {
                "id": j.id,
                "dim": j.dim,
                "anchor_slice": j.anchor_slice,
                "anchor_bar_id": j.anchor_bar.get("id"),
                "status": j.status,
                "current_bar_id": None if j.current_bar is None else j.current_bar.get("id"),
                "events": j.events,
            }
        )

    return {
        "journeys": journeys_out,
        "transitions": transitions,
        "config": {
            "lambda_sem": lambda_sem,
            "epsilon_match": epsilon_match,
            "theta_carry": theta_carry,
            "delta_sem_max": delta_sem_max,
            "topo_endpoint_eps": topo_endpoint_eps,
        },
    }
