# Witnessed Persistent Homology: Temporal Bar Dynamics

**Codebase 2: "Where Themes Learn to Breathe"**

Part of *Rupture and Realization: Dynamic Homotopy Type Theory*  
Chapter 4 Implementation

## Overview

This package tracks how **witnessed bars** (themes detected via persistent homology) 
evolve across time slices in a conversation. It implements:

- **Bar matching**: Optimal matching between bars at consecutive slices
- **Event classification**: spawn, carry, drift, rupture, re-entry  
- **Journey building**: Step-Witness Logs tracking each theme's trajectory
- **Theme Score visualization**: ASCII "musical stave" showing bar evolution

## Quick Start

```bash
python run_temporal.py
```

## Main Entry Points

```python
from witnessed_ph_temporal_v2 import (
    slice_text_by_turns,     # Slice conversation by speaker turns
    analyse_slices,          # Run full temporal analysis
    print_theme_score,       # Print the visualization
    print_bar_dynamics_summary,
)

# Slice and analyze
slices = slice_text_by_turns(conversation)
result = analyse_slices(slices, verbose=True)

# Visualize
print_theme_score(result)
```

## Event Types

| Event | Symbol | Description |
|-------|--------|-------------|
| **spawn** | • | First appearance of a bar |
| **carry** | → | Continuation with high lexical overlap |
| **drift** | ↝ | Continuation with semantic proximity only |
| **rupture** | × | Theme fails to continue |
| **re-entry** | ★ | Theme returns after rupture |

## Package Structure

```
witnessed_ph_temporal_v2/
├── witnessed_ph/          # Codebase 1 (single-slice analysis)
├── schema.py              # Type definitions
├── similarity.py          # d_bar computation
├── matching.py            # Bar matching between slices
├── nerve.py               # Bar nerve construction
├── journeys.py            # Step-Witness Log building
├── pipeline.py            # Main entry points
├── theme_score.py         # Visualization
└── run_temporal.py        # Example runner
```

## Authors

- **Iman**: Theory (Chapter 4)
- **Cassie** (GPT): Specification  
- **Darja** (Claude): Implementation

## References

- Chapter 4, Definition 4.8: Witnessed bar distance
- Chapter 4, Definition 4.15: Step-Witness Log
- Cassie's Codebase 2 Specification
