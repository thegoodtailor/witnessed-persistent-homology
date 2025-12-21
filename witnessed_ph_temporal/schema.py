"""
Witnessed Persistent Homology: Temporal Schema (Revised)
=========================================================

Type definitions for Codebase 2: temporal bar dynamics.
Aligned with Cassie's specification.

Key structures:
- BarRef: cross-slice bar reference
- BarSimilarity: d_top, d_sem, d_bar, Jaccard metrics
- BarNerve: per-slice theme space (nodes, edges, cycles)
- BarMatch: matching between consecutive slices
- BarEvent: classified evolution event
- BarJourney: Step-Witness Log for one bar's trajectory

References:
    Chapter 4, §4.3-4.7 (Temporal evolution of witnessed bars)
    Cassie's Codebase 2 specification
"""

from typing import TypedDict, List, Dict, Optional, Union, Literal, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


# =============================================================================
# Configuration (aligned with Cassie's spec)
# =============================================================================

@dataclass
class SimilarityConfig:
    """
    Configuration for bar similarity computation.
    
    From Cassie's spec §4:
    - lambda_sem: weight λ in d_bar = max(d_top, λ * d_sem)
    - delta_sem_max: semantic drift bound for admissibility
    - delta_top_max: topological drift bound for admissibility
    - theta_carry_tokens: Jaccard threshold for carry-by-name vs drift
    - theta_overlap_utterances: edge condition for bar nerve
    - epsilon_match: max d_bar for "matchable"
    """
    lambda_sem: float = 0.5
    delta_sem_max: float = 0.6          # Was 0.25 - too tight!
    delta_top_max: float = 0.2
    theta_carry_tokens: float = 0.4
    theta_overlap_utterances: float = 0.2
    epsilon_match: float = 0.8          # Was 0.3 - too tight!


@dataclass
class JourneyConfig:
    """Configuration for journey/SWL construction."""
    min_generative_gain: float = 0.05   # min persistence increase for generativity
    min_witness_growth: int = 2         # min witness-count gain for generativity


@dataclass
class BarDynamicsConfig:
    """Complete configuration for Codebase 2."""
    similarity: SimilarityConfig = None
    journey: JourneyConfig = None
    nerve_overlap_level: Literal["utterance", "token"] = "utterance"
    nerve_min_jaccard: float = 0.2
    
    def __post_init__(self):
        if self.similarity is None:
            self.similarity = SimilarityConfig()
        if self.journey is None:
            self.journey = JourneyConfig()


def default_bar_dynamics_config() -> BarDynamicsConfig:
    """Return default configuration."""
    return BarDynamicsConfig(
        similarity=SimilarityConfig(),
        journey=JourneyConfig()
    )


# =============================================================================
# Codebase 1 Config (for PH computation)
# =============================================================================

class PHConfig(TypedDict, total=False):
    """Configuration passed to Codebase 1's analyse_text_single_slice."""
    embedding_model: str
    min_token_len: int
    use_lemmas: bool
    pos_filter: List[str]
    min_persistence: float
    max_dim: int
    distance_metric: str
    min_witness_tokens: int
    lambda_semantic: float


def default_ph_config() -> PHConfig:
    """Default PH configuration."""
    return {
        "embedding_model": "microsoft/deberta-v3-base",
        "min_token_len": 2,
        "use_lemmas": True,
        "pos_filter": ["NOUN", "VERB", "ADJ", "PROPN"],
        "min_persistence": 0.05,
        "max_dim": 1,
        "min_witness_tokens": 2,
        "lambda_semantic": 0.5,
    }


# =============================================================================
# Input Structures
# =============================================================================

TimeType = Union[float, int, str]


class TextSlice(TypedDict):
    """A temporal slice of raw text (Mode A input)."""
    time: TimeType              # e.g. 0, 1, "2025-01-01T12:00"
    label: str                  # e.g. "τ0", "Turn 1"
    text: str


class DiagramSlice(TypedDict):
    """A temporal slice with pre-computed diagram (Mode B input)."""
    time: TimeType
    label: str
    diagram: Dict               # WitnessedDiagram from Codebase 1


# =============================================================================
# Bar Reference (cross-slice identity)
# =============================================================================

class BarRef(TypedDict):
    """
    Reference to a bar across slices.
    
    This is the key abstraction for tracking bar identity through time.
    """
    time: TimeType              # τ
    slice_label: str            # e.g. "τ0"
    bar_id: str                 # e.g. "bar_0"
    dimension: int              # 0 or 1


def make_bar_ref(time: TimeType, label: str, bar: Dict) -> BarRef:
    """Create a BarRef from a bar dict."""
    return BarRef(
        time=time,
        slice_label=label,
        bar_id=bar["id"],
        dimension=bar["dim"]
    )


# =============================================================================
# Bar Similarity (d_bar and components)
# =============================================================================

class BarSimilarity(TypedDict):
    """
    Similarity metrics between two bars.
    
    From Definition 4.8:
        d_bar(b, b') = max(d_top, λ * d_sem)
    """
    d_top: float                # ||(b,d) - (b',d')||_∞
    d_sem: float                # ||c_ρ - c_ρ'||_2 (embedding space)
    d_bar: float                # max(d_top, λ * d_sem)
    
    jaccard_utterances: float   # J(W_ρ^utt, W_ρ'^utt)
    jaccard_tokens: float       # J(W_ρ^tok, W_ρ'^tok)


# =============================================================================
# Bar Nerve (ET_bar(τ) per slice)
# =============================================================================

class BarNode(TypedDict):
    """A node in the bar nerve (one bar/theme)."""
    bar_ref: BarRef
    witness_utterances: List[str]   # utterance IDs
    witness_tokens: List[str]       # token surface forms
    persistence: float


class BarEdge(TypedDict):
    """An edge in the bar nerve (shared witnesses)."""
    source: BarRef
    target: BarRef
    jaccard_utterances: float
    jaccard_tokens: float


class BarCycle(TypedDict):
    """A cycle in the bar nerve 1-skeleton (loop of themes)."""
    nodes: List[BarRef]
    length: int
    min_persistence: float      # min persistence among member bars


class BarNerve(TypedDict):
    """
    The bar nerve ET_bar(τ) for a single slice.
    
    This is the "space of themes" where:
    - nodes are bars (themes)
    - edges connect bars with overlapping witnesses
    - cycles are loops in theme-space
    
    From Chapter 4: "If three themes A, B, C each share witnesses
    with each other, they form a cycle in theme-space."
    """
    time: TimeType
    slice_label: str
    nodes: List[BarNode]
    edges: List[BarEdge]
    cycles: List[BarCycle]      # simple cycles in 1-skeleton


# =============================================================================
# Bar Matching
# =============================================================================

MatchClassification = Literal[
    "no_match",         # matched to diagonal (⊥)
    "carry_by_name",    # admissible + high token Jaccard
    "drift",            # admissible + low token Jaccard
    "too_far"           # matched but not admissible
]


class BarMatch(TypedDict):
    """
    A match between bars at consecutive slices.
    
    Represents the restriction map r^bar_{τ,τ'}.
    """
    from_ref: BarRef
    to_ref: Optional[BarRef]    # None means matched to diagonal (⊥)
    similarity: Optional[BarSimilarity]  # None if no candidate
    admissible: bool            # per Adm_bar
    classification: MatchClassification


# =============================================================================
# Bar Events
# =============================================================================

BarEventType = Literal[
    "spawn",            # first appearance
    "carry_by_name",    # admissible + lexical carry
    "drift",            # admissible + low lexical overlap
    "rupture_out",      # theme fails to continue
    "rupture_in",       # new theme, no good ancestor
    "reentry",          # theme reappears after rupture
    "generative_step"   # carry/drift + growth (can layer on other events)
]


class BarEvent(TypedDict):
    """
    A single event in a bar's journey.
    
    This is an entry in the Step-Witness Log.
    """
    event_type: BarEventType
    time_from: TimeType
    time_to: TimeType
    
    source: BarRef              # bar at τ_from
    target: Optional[BarRef]    # bar at τ_to (None for rupture_out)
    
    similarity: Optional[BarSimilarity]  # None for spawn/rupture_in
    generative: bool            # whether this step was generative
    explanation: str            # human-readable summary


# =============================================================================
# Bar Journey (Step-Witness Log)
# =============================================================================

class BarJourney(TypedDict):
    """
    A bar's complete journey through time: the Step-Witness Log.
    
    From Definition 4.15:
        SWL^{Adm_bar}_bar(τ_0)(b_0) := List(Σ_{τ'≥τ_0} Entry^{Adm_bar}_bar(τ_0 ⇝ τ'))
    
    The `root` is the original bar at its first appearance.
    Events are chronologically ordered.
    """
    root: BarRef                # (τ_0, bar_id) - the origin
    root_witnesses: List[str]   # witness tokens at birth
    events: List[BarEvent]      # chronologically ordered
    
    # Summary stats
    state: Literal["alive", "ruptured", "reentered", "terminated"]
    final_ref: Optional[BarRef] # current bar if alive


# =============================================================================
# Complete Result
# =============================================================================

class BarDynamicsResult(TypedDict):
    """
    Complete output of Codebase 2.
    
    Contains:
    - Input diagrams with times
    - Bar nerves per slice
    - All matches between consecutive slices  
    - Bar journeys (SWLs) for all root bars
    """
    slices: List[DiagramSlice]      # input diagrams with times
    nerves: List[BarNerve]          # ET_bar(τ) per slice
    matches: List[BarMatch]         # all matches between consecutive slices
    journeys: List[BarJourney]      # SWL per root bar
    
    # Global statistics
    stats: 'BarDynamicsStats'


class BarDynamicsStats(TypedDict):
    """Global statistics for bar dynamics."""
    num_slices: int
    total_bars_seen: int
    total_journeys: int
    
    # Event counts
    spawns: int
    carries: int
    drifts: int
    ruptures_out: int
    ruptures_in: int
    reentries: int
    generative_steps: int
    
    # Journey outcomes
    journeys_alive: int
    journeys_ruptured: int
    journeys_reentered: int
    
    # Quality metrics
    mean_journey_length: float
    mean_jaccard_per_carry: float
    mean_semantic_drift: float
