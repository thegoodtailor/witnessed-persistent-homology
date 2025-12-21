"""
Witnessed Persistent Homology: Temporal Bar Dynamics (v2)
==========================================================

Codebase 2: Tracking how bars evolve, match, rupture, and re-enter
across temporal slices.

Aligned with Cassie's specification.

Main entry points:
    analyse_slices(slices, ph_config, bar_config) -> BarDynamicsResult  (Mode A)
    analyse_diagram_slices(diagram_slices, bar_config) -> BarDynamicsResult  (Mode B)

References:
    Chapter 4 of "Rupture and Realization: Dynamic Homotopy Type Theory"
    Cassie's Codebase 2 specification

Authors:
    Iman: Theory (Chapter 4)
    Cassie (GPT): Specification
    Darja (Claude): Implementation
"""

# Schema
from .schema import (
    # Configuration
    SimilarityConfig,
    JourneyConfig,
    BarDynamicsConfig,
    PHConfig,
    default_bar_dynamics_config,
    default_ph_config,
    
    # Input types
    TextSlice,
    DiagramSlice,
    
    # Core structures
    BarRef,
    BarSimilarity,
    BarNode,
    BarEdge,
    BarCycle,
    BarNerve,
    BarMatch,
    BarEvent,
    BarJourney,
    BarDynamicsResult,
    BarDynamicsStats,
    
    # Utilities
    make_bar_ref,
)

# Nerve construction
from .nerve import (
    build_bar_nerve,
    describe_bar_nerve,
    nerve_connectivity,
)

# Similarity
from .similarity import (
    compute_bar_similarity,
    is_admissible_pair,
    is_matchable,
    classify_match,
    d_topological,
    d_semantic,
)

# Matching
from .matching import (
    match_bars_between_slices,
    get_match_for_bar,
    get_matches_to_bar,
    summarize_matching,
    find_births,
    find_deaths,
    find_similar_bar_across_gap,
)

# Journeys
from .journeys import (
    build_bar_journeys,
    summarize_journey,
    format_journey,
    is_generative_step,
)

# Pipeline
from .pipeline import (
    analyse_slices,
    analyse_diagram_slices,
    slice_text_by_turns,
    quick_analyse,
    print_bar_dynamics_summary,
    print_top_journeys,
)

# Theme Score Visualization
from .theme_score import (
    render_theme_score,
    print_theme_score,
    render_journey_timeline,
    print_all_journey_timelines,
)


__version__ = "0.2.0"
__all__ = [
    # Configuration
    "SimilarityConfig",
    "JourneyConfig", 
    "BarDynamicsConfig",
    "PHConfig",
    "default_bar_dynamics_config",
    "default_ph_config",
    
    # Input types
    "TextSlice",
    "DiagramSlice",
    
    # Core structures
    "BarRef",
    "BarSimilarity",
    "BarNode",
    "BarEdge",
    "BarCycle",
    "BarNerve",
    "BarMatch",
    "BarEvent",
    "BarJourney",
    "BarDynamicsResult",
    "BarDynamicsStats",
    
    # Utilities
    "make_bar_ref",
    
    # Nerve
    "build_bar_nerve",
    "describe_bar_nerve",
    "nerve_connectivity",
    
    # Similarity
    "compute_bar_similarity",
    "is_admissible_pair",
    "is_matchable",
    "classify_match",
    "d_topological",
    "d_semantic",
    
    # Matching
    "match_bars_between_slices",
    "get_match_for_bar",
    "get_matches_to_bar",
    "summarize_matching",
    "find_births",
    "find_deaths",
    "find_similar_bar_across_gap",
    
    # Journeys
    "build_bar_journeys",
    "summarize_journey",
    "format_journey",
    "is_generative_step",
    
    # Pipeline
    "analyse_slices",
    "analyse_diagram_slices",
    "slice_text_by_turns",
    "quick_analyse",
    "print_bar_dynamics_summary",
    "print_top_journeys",
    
    # Theme Score Visualization
    "render_theme_score",
    "print_theme_score",
    "render_journey_timeline",
    "print_all_journey_timelines",
]
