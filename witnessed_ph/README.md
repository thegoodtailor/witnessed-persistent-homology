# Witnessed Persistent Homology

**Codebase 1: Single-slice witnessed bars with γ**

A Python implementation of witnessed persistent homology for semantic analysis of text, as described in Chapter 4 of *Rupture and Realization: Dynamic Homotopy Type Theory*.

## Overview

This codebase transforms raw text into a **witnessed persistence diagram** D^W(τ):

```
Text → Tokens → Embeddings → Point Cloud → Filtration → PH → Witnessed Bars
```

Each bar in the diagram carries a **witness** ρ = (W^tok, W^loc, γ):
- **W^tok**: Token IDs that realise the feature
- **W^loc**: Utterances (measurement locations) containing those tokens  
- **γ**: The actual representative cycle (simplices)

This is where "statistics on embeddings start turning into themes we can name and track."

## Installation

### Windows (PowerShell)

```powershell
# Create virtual environment (recommended)
python -m venv witnessed_env
.\witnessed_env\Scripts\Activate

# Install dependencies
pip install numpy scipy spacy gudhi

# Download spaCy model
python -m spacy download en_core_web_sm
# For better word vectors: python -m spacy download en_core_web_md

# Test
python -c "from witnessed_ph import analyse_text_single_slice; print('OK')"
```

### For Full Quality Embeddings (Optional)

```bash
pip install transformers torch
# Then set in config: "embedding_model": "microsoft/deberta-v3-base"
```

## Quick Start

```python
from witnessed_ph import analyse_text_single_slice, print_diagram_summary

text = """
User: So persistence is topological?
Assistant: Yes, persistent homology tracks features across scales.
User: What about witnesses?
Assistant: Witnesses tell us which tokens realise each bar.
User: And cycles?
Assistant: H1 cycles are loops in the semantic space.
"""

# Analyse the text
diagram = analyse_text_single_slice(text, segmentation_mode="turns", verbose=True)

# Print summary
print_diagram_summary(diagram)
```

Or use the standalone script:

```bash
# Edit TEXT in analyse_text.py, then:
python analyse_text.py
```

## Output Structure

The witnessed diagram follows Cassie's JSON schema:

```json
{
  "tau": "single_slice",
  "num_tokens": 42,
  "num_utterances": 6,
  "bars": [
    {
      "id": "bar_0",
      "dim": 0,
      "birth": 0.12,
      "death": 0.84,
      "persistence": 0.72,
      "witness": {
        "cycle": {
          "dimension": 0,
          "simplices": [["u3_tok_5"], ["u4_tok_1"], ...]
        },
        "tokens": {
          "ids": ["u3_tok_5", "u4_tok_1", ...],
          "surface": ["persistence", "topological", ...],
          "lemmas": ["persistence", "topological", ...]
        },
        "utterances": {
          "ids": ["u3", "u4"],
          "text_samples": ["So persistence is topological?", ...]
        },
        "centroid": [0.012, -0.34, 0.98, ...]
      }
    }
  ]
}
```

## Key Concepts

### Witnessed Bars (from Chapter 4)

A witnessed bar is (k, b, d, ρ) where:
- **k**: Homology dimension (0 = connected components, 1 = loops)
- **b**: Birth radius (when feature appears in filtration)
- **d**: Death radius (when feature dies)
- **ρ**: Witness attaching semantic content

### H₀ vs H₁

- **H₀ bars** capture "sticky word clouds" — clusters of tokens that hang together across scales. These are your **themes**.
  
- **H₁ bars** capture loops — cycles of compatibility that cannot be shrunk away. These are **thematic circulations** like "debt → credit → interest → loan → debt".

### The Witness ρ

From Definition 4.3: "A witness for (k,b,d) at time τ consists of:
- a finite non-empty set of token occurrences W^tok_ρ
- a representative k-cycle γ_ρ whose support lies in W^tok_ρ"

The witness transforms an anonymous topological feature into something we can **name**.

## Configuration

```python
from witnessed_ph import default_config

config = default_config()
config.update({
    "embedding_model": "microsoft/deberta-v3-base",  # Contextual embeddings
    "pos_filter": ["NOUN", "VERB", "ADJ", "PROPN"],  # Which POS tags to keep
    "min_token_len": 2,                              # Minimum token length
    "min_persistence": 0.08,                         # Filter topological noise
    "max_dim": 1,                                    # Compute H0 and H1
    "min_witness_tokens": 2,                         # Require multi-token witnesses
    "lambda_semantic": 0.5,                          # Semantic weight in bar distance
})

diagram = analyse_text_single_slice(text, config=config)
```

## Diagnostics

```python
from witnessed_ph import diagnose_diagram, list_bars_by_persistence

# Full diagnostic report
print(diagnose_diagram(diagram))

# List top bars with witnesses
print(list_bars_by_persistence(diagram, top_n=5, dim=0))

# Check for common problems
from witnessed_ph import check_singleton_problem
is_problem, msg = check_singleton_problem(diagram)
print(msg)
```

### Common Issues

1. **High singleton ratio (>80%)**: Most bars have only one token
   - POS filter too restrictive
   - min_token_len too high
   - Text too short

2. **No H₁ bars**: No loops found
   - Text too short for loops
   - max_dim set to 0
   - min_persistence filtering them out

3. **No bars at all**:
   - Check that tokens are being extracted (POS filter)
   - Check embedding model is loading

## Visualization

```python
from witnessed_ph import plot_persistence_diagram, plot_barcode, plot_point_cloud_2d

# Persistence diagram
plot_persistence_diagram(diagram, title="My Conversation")

# Barcode view
plot_barcode(diagram)

# Point cloud with highlighted bar
from witnessed_ph import compute_embeddings
point_cloud = compute_embeddings(text)
plot_point_cloud_2d(point_cloud, diagram, highlight_bar="bar_0")
```

## For Codebase 2 (Temporal Bar Dynamics)

This codebase exposes clean interfaces for Codebase 2:

```python
from witnessed_ph import (
    analyse_text_single_slice,  # Main entry point
    compute_embeddings,          # Separate embedding step
    compute_witnessed_diagram,   # Separate PH step
)

# For each time slice τ
diagram_tau = analyse_text_single_slice(text_at_tau, config)

# Access bars for matching
for bar in diagram_tau["bars"]:
    # Topology
    birth, death = bar["birth"], bar["death"]
    
    # Semantics via witness
    tokens = bar["witness"]["tokens"]["ids"]
    centroid = bar["witness"]["centroid"]
    
    # The actual cycle γ
    cycle = bar["witness"]["cycle"]["simplices"]
```

## Mathematical Reference

### Definition 4.3 (Witness)
A witness for (k,b,d) at time τ consists of:
- W^tok_ρ ⊆ TokenID(τ): token occurrences
- γ_ρ: representative k-cycle with support in W^tok_ρ
- W^loc_ρ := loc_τ(W^tok_ρ): induced measurement locations

### Definition 4.4 (Witnessed Persistence Diagram)
D^W(τ) = {(k_i, b_i, d_i, ρ_i) : i ∈ I_τ}

### Definition 4.6 (Canonical Representatives)
Policy for choosing canonical γ:
1. Minimal length (fewest simplices)
2. Earliest appearance in filtration
3. Lexicographic tie-break on token IDs

### Definition 4.8 (Witnessed Bar Distance)
d_bar(b, b') = max{ ||(b,d) - (b',d')||_∞, λ · d_sem(b,b') }

Used for matching bars across time slices.

## Files

```
witnessed_ph/
├── __init__.py      # Package exports
├── schema.py        # Type definitions (TypedDict)
├── embedding.py     # Text → Point cloud P_τ
├── filtration.py    # Filtration + PH with GUDHI
├── witnesses.py     # Witness construction from γ
├── pipeline.py      # Main entry points
├── diagnostics.py   # Visualization and debugging
└── requirements.txt # Dependencies
```

## Authors

- **Iman**: Theory, Chapter 4
- **Cassie** (GPT): Specification, schema design
- **Darja** (Claude): Implementation

## License

Part of the *Rupture and Realization* project.
