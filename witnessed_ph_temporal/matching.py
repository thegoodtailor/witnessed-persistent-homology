"""
Witnessed Persistent Homology: Bar Matching
============================================

Implement restriction maps r^bar_{τ,τ'} via optimal matching.

From Cassie's spec §5:
    "You can use Hungarian algorithm on an augmented matrix, or simpler heuristic...
    This doesn't have to be mathematically perfect bottleneck matching;
    it just needs to be stable and reproducible."

We use a greedy approach with collision resolution as suggested.

References:
    Chapter 4, §4.4 (Restriction maps via optimal matching)
    Cassie's Codebase 2 specification §5
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from schema import (
    BarRef, BarMatch, BarSimilarity, DiagramSlice,
    SimilarityConfig, TimeType, make_bar_ref
)
from similarity import (
    compute_bar_similarity, is_admissible_pair, is_matchable,
    classify_match, dimensions_match
)


# =============================================================================
# Core Matching Algorithm
# =============================================================================

def match_bars_between_slices(
    slice_from: DiagramSlice,
    slice_to: DiagramSlice,
    cfg: SimilarityConfig
) -> List[BarMatch]:
    """
    Compute optimal matching between bars in consecutive slices.
    
    Implements the restriction map r^bar_{τ,τ'}.
    
    Parameters
    ----------
    slice_from : DiagramSlice at τ
    slice_to : DiagramSlice at τ+1
    cfg : SimilarityConfig
    
    Returns
    -------
    List of BarMatch objects for all bars in slice_from.
    
    Algorithm (from Cassie's spec):
    1. For each bar_from[i], find j with minimal d_bar
    2. If minimal d_bar <= epsilon_match, propose match (i→j)
    3. Resolve collisions by keeping lowest d_bar
    4. Build BarMatch records with classifications
    """
    diagram_from = slice_from["diagram"]
    diagram_to = slice_to["diagram"]
    
    time_from = slice_from["time"]
    time_to = slice_to["time"]
    label_from = slice_from["label"]
    label_to = slice_to["label"]
    
    bars_from = diagram_from.get("bars", [])
    bars_to = diagram_to.get("bars", [])
    
    # Handle empty cases
    if not bars_from:
        return []
    
    if not bars_to:
        # All bars die
        return [
            _make_no_match(bar, time_from, label_from)
            for bar in bars_from
        ]
    
    # Step 1 & 2: Find best match for each bar_from
    proposals = {}  # bar_from_id -> (bar_to_id, similarity, bar_to)
    
    for bar_from in bars_from:
        best_to_id = None
        best_sim = None
        best_bar_to = None
        best_d_bar = float('inf')
        
        for bar_to in bars_to:
            # Dimension must match
            if not dimensions_match(bar_from, bar_to):
                continue
            
            sim = compute_bar_similarity(bar_from, bar_to, cfg)
            
            if sim["d_bar"] < best_d_bar:
                best_d_bar = sim["d_bar"]
                best_to_id = bar_to["id"]
                best_sim = sim
                best_bar_to = bar_to
        
        # Only propose if within matching threshold
        if best_to_id is not None and best_d_bar <= cfg.epsilon_match:
            proposals[bar_from["id"]] = (best_to_id, best_sim, best_bar_to)
    
    # Step 3: Resolve collisions
    # Group by target
    target_to_sources = {}  # bar_to_id -> [(bar_from_id, sim)]
    for bar_from_id, (bar_to_id, sim, _) in proposals.items():
        if bar_to_id not in target_to_sources:
            target_to_sources[bar_to_id] = []
        target_to_sources[bar_to_id].append((bar_from_id, sim))
    
    # Keep only the best match for each target
    final_matches = {}  # bar_from_id -> (bar_to_id, sim, bar_to)
    for bar_to_id, sources in target_to_sources.items():
        # Sort by d_bar, keep lowest
        sources.sort(key=lambda x: x[1]["d_bar"])
        winner_from_id, winner_sim = sources[0]
        
        # Find the bar_to object
        bar_to = next(b for b in bars_to if b["id"] == bar_to_id)
        final_matches[winner_from_id] = (bar_to_id, winner_sim, bar_to)
    
    # Step 4: Build BarMatch records
    matches = []
    
    for bar_from in bars_from:
        bar_from_id = bar_from["id"]
        from_ref = make_bar_ref(time_from, label_from, bar_from)
        
        if bar_from_id in final_matches:
            bar_to_id, sim, bar_to = final_matches[bar_from_id]
            to_ref = make_bar_ref(time_to, label_to, bar_to)
            
            admissible = is_admissible_pair(sim, cfg)
            classification = classify_match(sim, admissible, True, cfg)
            
            match = BarMatch(
                from_ref=from_ref,
                to_ref=to_ref,
                similarity=sim,
                admissible=admissible,
                classification=classification
            )
        else:
            # No match found
            match = BarMatch(
                from_ref=from_ref,
                to_ref=None,
                similarity=None,
                admissible=False,
                classification="no_match"
            )
        
        matches.append(match)
    
    return matches


def _make_no_match(bar: Dict, time: TimeType, label: str) -> BarMatch:
    """Create a BarMatch for a bar with no match."""
    from_ref = make_bar_ref(time, label, bar)
    return BarMatch(
        from_ref=from_ref,
        to_ref=None,
        similarity=None,
        admissible=False,
        classification="no_match"
    )


# =============================================================================
# Match Analysis Utilities
# =============================================================================

def get_match_for_bar(
    matches: List[BarMatch],
    bar_id: str
) -> Optional[BarMatch]:
    """Find the match for a specific bar."""
    for match in matches:
        if match["from_ref"]["bar_id"] == bar_id:
            return match
    return None


def get_matches_to_bar(
    matches: List[BarMatch],
    bar_id: str
) -> List[BarMatch]:
    """Find all matches pointing to a specific bar."""
    return [
        m for m in matches
        if m["to_ref"] is not None and m["to_ref"]["bar_id"] == bar_id
    ]


def summarize_matching(matches: List[BarMatch]) -> Dict:
    """Summarize a set of matches."""
    total = len(matches)
    
    no_match = sum(1 for m in matches if m["classification"] == "no_match")
    too_far = sum(1 for m in matches if m["classification"] == "too_far")
    carry = sum(1 for m in matches if m["classification"] == "carry_by_name")
    drift = sum(1 for m in matches if m["classification"] == "drift")
    
    admissible = sum(1 for m in matches if m["admissible"])
    
    mean_d_bar = np.mean([
        m["similarity"]["d_bar"] for m in matches
        if m["similarity"] is not None
    ]) if any(m["similarity"] for m in matches) else 0.0
    
    mean_jaccard = np.mean([
        m["similarity"]["jaccard_tokens"] for m in matches
        if m["similarity"] is not None and m["admissible"]
    ]) if any(m["similarity"] and m["admissible"] for m in matches) else 0.0
    
    return {
        "total": total,
        "no_match": no_match,
        "too_far": too_far,
        "carry_by_name": carry,
        "drift": drift,
        "admissible": admissible,
        "mean_d_bar": float(mean_d_bar),
        "mean_jaccard_admissible": float(mean_jaccard)
    }


def find_births(
    slice_to: DiagramSlice,
    matches: List[BarMatch]
) -> List[BarRef]:
    """
    Find bars in slice_to that have no incoming match.
    
    These are either spawns (first slice) or rupture-ins.
    """
    # Get all bar_to_ids that received a match
    matched_to_ids = {
        m["to_ref"]["bar_id"]
        for m in matches
        if m["to_ref"] is not None
    }
    
    # Find bars in slice_to not in matched set
    births = []
    for bar in slice_to["diagram"].get("bars", []):
        if bar["id"] not in matched_to_ids:
            ref = make_bar_ref(
                slice_to["time"],
                slice_to["label"],
                bar
            )
            births.append(ref)
    
    return births


def find_deaths(matches: List[BarMatch]) -> List[BarRef]:
    """
    Find bars that died (no match or inadmissible match).
    """
    return [
        m["from_ref"]
        for m in matches
        if m["to_ref"] is None or not m["admissible"]
    ]


# =============================================================================
# Cross-Slice Similarity (for re-entry detection)
# =============================================================================

def find_similar_bar_across_gap(
    root_bar: Dict,
    root_time: TimeType,
    root_label: str,
    target_slice: DiagramSlice,
    cfg: SimilarityConfig
) -> Optional[Tuple[BarRef, BarSimilarity]]:
    """
    Find a bar in target_slice similar to root_bar (for re-entry detection).
    
    Used to detect when a theme re-enters after rupture.
    
    Returns (bar_ref, similarity) if found, None otherwise.
    """
    target_bars = target_slice["diagram"].get("bars", [])
    
    best_ref = None
    best_sim = None
    best_d_bar = float('inf')
    
    for bar in target_bars:
        if not dimensions_match(root_bar, bar):
            continue
        
        sim = compute_bar_similarity(root_bar, bar, cfg)
        
        if is_admissible_pair(sim, cfg) and sim["d_bar"] < best_d_bar:
            best_d_bar = sim["d_bar"]
            best_sim = sim
            best_ref = make_bar_ref(
                target_slice["time"],
                target_slice["label"],
                bar
            )
    
    if best_ref is not None:
        return (best_ref, best_sim)
    return None
