"""
Witnessed Persistent Homology: Bar Similarity
==============================================

Compute d_bar and related similarity metrics between bars.

Key insight from Cassie's spec: admissibility is ONLY about bounds
(d_top ≤ Δ, d_sem ≤ δ). Classification into carry vs drift happens
AFTER admissibility using Jaccard.

From Definition 4.8:
    d_bar(b, b') = max{ d_top(b, b'), λ · d_sem(b, b') }

References:
    Chapter 4, §4.3 (Witnessed bar distance)
    Cassie's Codebase 2 specification §4
"""

from typing import Dict, Set, Tuple
import numpy as np

from schema import (
    BarSimilarity, SimilarityConfig, MatchClassification
)


# =============================================================================
# Jaccard Utility
# =============================================================================

def jaccard(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard index between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# Distance Components
# =============================================================================

def d_topological(bar_a: Dict, bar_b: Dict) -> float:
    """
    Topological distance between bars.
    
    From Definition 4.8:
        d_top(b, b') = ||(b, d) - (b', d')||_∞
    
    Uses L^∞ norm on (birth, death) pairs.
    """
    birth_a, death_a = bar_a["birth"], bar_a["death"]
    birth_b, death_b = bar_b["birth"], bar_b["death"]
    
    # Handle infinite death (essential bars)
    if np.isinf(death_a) and np.isinf(death_b):
        return abs(birth_a - birth_b)
    elif np.isinf(death_a) or np.isinf(death_b):
        return float('inf')  # Can't match finite with infinite
    
    return max(abs(birth_a - birth_b), abs(death_a - death_b))


def d_semantic(bar_a: Dict, bar_b: Dict) -> float:
    """
    Semantic distance between bars via witness centroids.
    
    From Definition 4.8:
        d_sem(b, b') = ||c_ρ - c_ρ'||_2
    """
    c_a = np.array(bar_a["witness"]["centroid"])
    c_b = np.array(bar_b["witness"]["centroid"])
    
    return float(np.linalg.norm(c_a - c_b))


# =============================================================================
# Core Similarity Computation
# =============================================================================

def compute_bar_similarity(
    bar_a: Dict,
    bar_b: Dict,
    cfg: SimilarityConfig
) -> BarSimilarity:
    """
    Compute all similarity metrics between two bars.
    
    From Cassie's spec §4:
        d_bar = max(d_top, λ * d_sem)
    
    Parameters
    ----------
    bar_a, bar_b : Bar dicts from Codebase 1
    cfg : SimilarityConfig
    
    Returns
    -------
    BarSimilarity with all distance and overlap metrics.
    """
    # Topological distance
    d_top = d_topological(bar_a, bar_b)
    
    # Semantic distance
    d_sem = d_semantic(bar_a, bar_b)
    
    # Combined distance
    d_bar = max(d_top, cfg.lambda_sem * d_sem)
    
    # Lexical overlaps
    witness_a = bar_a.get("witness", {})
    witness_b = bar_b.get("witness", {})
    
    toks_a = set(witness_a.get("tokens", {}).get("ids", []))
    toks_b = set(witness_b.get("tokens", {}).get("ids", []))
    
    utt_a = set(witness_a.get("utterances", {}).get("ids", []))
    utt_b = set(witness_b.get("utterances", {}).get("ids", []))
    
    j_tok = jaccard(toks_a, toks_b)
    j_utt = jaccard(utt_a, utt_b)
    
    return BarSimilarity(
        d_top=d_top,
        d_sem=d_sem,
        d_bar=d_bar,
        jaccard_tokens=j_tok,
        jaccard_utterances=j_utt
    )


# =============================================================================
# Admissibility (separate from classification!)
# =============================================================================

def is_admissible_pair(
    sim: BarSimilarity,
    cfg: SimilarityConfig
) -> bool:
    """
    Check if a bar pair satisfies admissibility criteria.
    
    From Cassie's spec §4 and Definition 4.11:
    Admissibility is ONLY about bounds:
        d_top ≤ delta_top_max  AND  d_sem ≤ delta_sem_max
    
    Classification (carry vs drift) is separate and uses Jaccard.
    """
    return (
        sim["d_top"] <= cfg.delta_top_max and
        sim["d_sem"] <= cfg.delta_sem_max
    )


def is_matchable(
    sim: BarSimilarity,
    cfg: SimilarityConfig
) -> bool:
    """
    Check if d_bar is within matching threshold.
    
    A bar can be "matchable" but not "admissible" if
    d_bar <= epsilon_match but bounds are exceeded.
    """
    return sim["d_bar"] <= cfg.epsilon_match


# =============================================================================
# Classification (happens AFTER admissibility check)
# =============================================================================

def classify_match(
    sim: BarSimilarity,
    admissible: bool,
    has_match: bool,
    cfg: SimilarityConfig
) -> MatchClassification:
    """
    Classify a match based on similarity and admissibility.
    
    From Cassie's spec §5:
        - no_match: matched to diagonal (⊥)
        - too_far: matched but not admissible
        - carry_by_name: admissible + high token Jaccard
        - drift: admissible + low token Jaccard
    
    Key insight: classification happens AFTER admissibility.
    """
    if not has_match:
        return "no_match"
    
    if not admissible:
        return "too_far"
    
    # Now we know it's admissible - classify by token overlap
    if sim["jaccard_tokens"] >= cfg.theta_carry_tokens:
        return "carry_by_name"
    else:
        return "drift"


# =============================================================================
# Dimension Check
# =============================================================================

def dimensions_match(bar_a: Dict, bar_b: Dict) -> bool:
    """
    Check if bars have the same homology dimension.
    
    From Definition 4.11: dimension preservation is required.
    """
    return bar_a.get("dim", 0) == bar_b.get("dim", 0)


# =============================================================================
# All-Pairs Similarity
# =============================================================================

def compute_all_pairwise_similarities(
    bars_from: list,
    bars_to: list,
    cfg: SimilarityConfig
) -> Dict[Tuple[str, str], BarSimilarity]:
    """
    Compute pairwise similarities between all bars in two lists.
    
    Returns dict mapping (bar_id_from, bar_id_to) to BarSimilarity.
    Only includes pairs with matching dimensions.
    """
    result = {}
    
    for bar_a in bars_from:
        for bar_b in bars_to:
            if not dimensions_match(bar_a, bar_b):
                continue
            
            sim = compute_bar_similarity(bar_a, bar_b, cfg)
            result[(bar_a["id"], bar_b["id"])] = sim
    
    return result
