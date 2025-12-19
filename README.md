# witnessed-ph

Reproducible, inspectable code for the **Chapter 4 “witnessed persistence / witnessed bars”** experiments.

The goal of this repo is not to be clever; it’s to be the thing you can:

- run today,
- run again next year,
- debug when something breaks,
- extend when you want a new experiment.

It includes **two closely-related pipelines**:

1. **Single-slice witnessed persistence** (static): turn one text slice τ into a witnessed diagram \(D^W(\tau)\).
2. **Temporal dynamics** (Chapter 4 “second experiment block”): treat each turn as a slice \(\tau_i\), compute \(D^W(\tau_i)\), then track bars as **carry / drift / rupture / re-entry / birth**.

---

## What models/libraries does this use?

- **Contextual embeddings:** `microsoft/deberta-v3-base` (HuggingFace Transformers) by default.
- **Tokenisation / POS filtering:** spaCy (`en_core_web_sm`) by default.
- **Persistent homology:**
  - **H0** is computed deterministically from the pairwise distance matrix (no external TDA dependency required).
  - **H1** (loops) is *optional* and uses **GUDHI** when installed.

If you only care about the Chapter 4 temporal “theme dynamics” experiments (H0), you can run without GUDHI.

---

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Windows PowerShell

```powershell
cd E:\GitHub\witnessed-persistent-homology

# Create + activate a venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional but recommended: install the package in editable mode
pip install -e .

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

> Notes
> - The first run will download the DeBERTa model weights into your HuggingFace cache.
> - If you want GPU, install a CUDA-enabled PyTorch build that matches your CUDA version.

### 2) Run the chapter scripts

```bash
python examples/chapter4_single_slice.py
python examples/chapter4_temporal_dynamics.py
```

### 3) Run tests

```bash
pytest -q
```

On some Windows setups, you may need to prefix with:

```powershell
$env:PYTHONPATH = "."
pytest -q
```

---

## What output should I expect?

### Single slice (static)

`examples/chapter4_single_slice.py` prints:

- a short **diagram summary** (token/utterance counts, number of bars)
- the **top H0 bars** by persistence, with their witness token list
- (optionally) the **top H1 bars**, if GUDHI is installed and loops appear

Typical console shape:

```text
[witnessed_ph] slice chapter4: 12 utterances
[witnessed_ph] loading embedding model: microsoft/deberta-v3-base

Diagram for slice chapter4
  tokens:     ~100
  utterances: 12
  bars:       ~20–40  (H0=~20–35, H1=~0–10)
  top bar:    h0_0 dim=0 [b, d) pers=p

========================================================================
Top H0 theme bars (by persistence):
  h0_0  [0.061, 0.231)  pers=0.170  witness=['climate', 'carbon', 'renewable', ...]
  h0_1  [0.074, 0.205)  pers=0.131  witness=['rome', 'empire', 'civilization', ...]
  ...

Top H1 loop bars (by persistence):
  h1_0  [0.146, 0.183)  pers=0.037  witness=['energy', 'change', 'carbon']  cycle_edges=[(a,b),(b,c),(a,c)]
  ...
```

The **exact numbers** depend on:

- your token filtering (POS, stopwords, min length)
- model version / tokenizer
- random seeds (mostly relevant if you change any sampling)
- CPU vs GPU numerical differences

But the *shape* of the output is stable: you’ll always get a list of bar dicts with birth/death/persistence and a witness payload.

### Temporal dynamics (the “second experiment block”)

`examples/chapter4_temporal_dynamics.py` prints:

- **per-transition event counts** (carry / drift / rupture / re-entry / birth)
- a book-style **Theme score (SWL) rendering** showing which themes spawn, carry, drift,
  rupture, and re-enter in each slice

```text
========================================================================
Temporal dynamics summary
========================================================================

Per-transition counts:
  τ0 → τ1 : carry=3, drift=1, rupture=2, reentry=0, birth=4
  τ1 → τ2 : carry=4, drift=2, rupture=1, reentry=1, birth=3
  ...

Totals:
  carry:   18
  drift:   11
  rupture:  7
  reentry:  3
  birth:   21

Config used:
  lambda_sem = 0.5
  epsilon_match = 0.8
  theta_carry = 0.4
  delta_sem_max = 0.6
  topo_endpoint_eps = 0.2

Theme score (SWL)
------------------------------------------------------------------------
Legend: • spawn   → carry   ~ drift   × rupture   F re-entry
...
```

Again, counts will vary, but the categories and output schema are stable.

---

## Why do some H0 bars have birth > 0?

This is the key “wait, what?” moment you ran into — and it’s not a bug.

### Raw H0 persistence (standard TDA)

In standard Vietoris–Rips persistence:

- every point starts as its own connected component,
- so **every H0 class is born at filtration value 0**.

If you compute *raw* H0 bars, births are always 0.

### Theme bars (Chapter 4 temporal work)

In the temporal experiments we generally do **not** want singleton “themes”.
A single token as a component is semantically uninteresting.

So we offer a *derived* H0 view:

- we keep the same single-linkage merge process,
- but we only treat a component as a “theme” once it reaches a minimum witness size (e.g. ≥ 2 tokens),
- and we define the **theme birth** as the first radius at which that component becomes non-trivial.

That “first radius where it becomes witnessable” is typically **> 0**.

In other words:

- **raw H0 bars:** births = 0 (components of individual points)
- **theme H0 bars:** births = first merge time that yields a multi-token component

Switch using `cfg["h0_mode"]`:

```python
cfg = default_config()
cfg["h0_mode"] = "raw"    # births=0
# cfg["h0_mode"] = "theme" # births can be >0
```

---

## Core API

### Analyse one slice

```python
from witnessed_ph import default_config, analyse_text_single_slice

text = """User: ...\nAssistant: ..."""

cfg = default_config()
cfg["min_persistence"] = 0.03
cfg["min_witness_tokens"] = 2
cfg["h0_mode"] = "theme"  # or "raw"

diagram = analyse_text_single_slice(text, config=cfg, segmentation_mode="lines", verbose=True)
```

### Analyse a conversation over time (turn-by-turn)

```python
from witnessed_ph.temporal import analyse_conversation_dynamics
from witnessed_ph import default_config

turns = [
  {"speaker": "User", "text": "..."},
  {"speaker": "Assistant", "text": "..."},
]

cfg = default_config()
result = analyse_conversation_dynamics(turns, config=cfg)

# result = {"diagrams": [...], "dynamics": {...}}
```

---

## Output schema

The “unit of truth” is a **diagram dict**:

```python
{
  "slice_id": "τ3",
  "num_tokens": 108,
  "num_utterances": 12,
  "config": {...},
  "embedding_model": "microsoft/deberta-v3-base",
  "distance_metric": "cosine",
  "library_versions": {...},
  "bars": [
     {
       "id": "h0_0",
       "dim": 0,
       "birth": 0.071,
       "death": 0.218,
       "persistence": 0.147,
       "witness": {
          "token_ids": ["t12", "t57", ...],
          "tokens": {
             "surface": ["climate", "carbon", ...],
             "lemma":   ["climate", "carbon", ...],
             "pos":     ["NOUN", "NOUN", ...]
          },
          "lemma_set": ["carbon", "climate", ...],
          "utterance_ids": ["u0", "u10"],
          "centroid": [...],
          # dim=1 only:
          "cycle": {"edges": [(0,1),(1,2),(0,2)]}
       }
     },
     ...
  ]
}
```

This is deliberately JSON-friendly.

---

## Algorithm details

### A. Single-slice witnessed persistence

Given a slice \(\tau\):

1. **Segment into utterances** (measurement locations): by lines/turns.
2. **Tokenise** with spaCy.
3. **Filter tokens** (default: content POS tags, min length, optional lemma mode).
4. **Embed each token occurrence** with DeBERTa (contextualised): each occurrence is a distinct point.
5. **Normalise** embeddings to the unit sphere.
6. **Compute pairwise distances** (default: angular/cosine distance in \([0,1]\)).
7. **Compute persistence**:
   - **H0 raw:** union–find over sorted edges to get births/deaths.
   - **H0 theme:** same union–find merge tree, but emit only clusters with witness size ≥ `min_witness_tokens`, with birth at the merge scale where the cluster becomes non-trivial.
   - **H1 (optional):** build a Vietoris–Rips complex (up to dim 2) in GUDHI and extract 1-dimensional bars.
8. **Witness extraction**:
   - H0: witness tokens are the cluster membership (truncated for readability), plus centroid.
   - H1: witness tokens are taken from the death simplex when available; if it’s a triangle we return its boundary edges as a concrete cycle.

### B. Temporal bar dynamics (Chapter 4 second block)

Given slice diagrams \(D^W(\tau_0),\dots,D^W(\tau_T)\):

1. For each consecutive pair \(\tau_t \to \tau_{t+1}\), match H0 bars.
2. Define a bar metric

   \[
   d_\text{bar}(a,b) = \max(\lVert (b_a,d_a)-(b_b,d_b)\rVert_\infty, \; \lambda\, d_\text{sem}(\mu_a,\mu_b))
   \]

   where \(\mu\) is the witness centroid and \(d_\text{sem}\) is angular distance.

3. A candidate match must satisfy:

   - `d_bar ≤ epsilon_match`
   - and endpoints separately stay within `topo_endpoint_eps` (optional tightening)
   - and semantic drift `sem ≤ delta_sem_max`

4. Classification:

   - **carry:** match exists AND witness lemma-set Jaccard ≥ `theta_carry`
   - **drift:** match exists but Jaccard < `theta_carry` (semantics still close)
   - **rupture:** no admissible match for an active journey
   - **re-entry:** a previously ruptured journey finds a later match (from its anchor)
   - **birth:** a current bar not claimed by any match

This implementation is intentionally explicit: it’s a practical, reproducible model of the calculus, with all thresholds surfaced in config.

---

## Configuration reference

Most-used knobs:

- `embedding_model`: HuggingFace model name
- `pos_filter`, `min_token_len`, `use_lemmas`
- `distance_metric`: `cosine` (angular) or `euclidean`
- `max_edge_length`: filtration truncation
- `min_persistence`: filter noise
- `min_witness_tokens`, `max_witness_tokens`
- `h0_mode`: `raw` or `theme`

Dynamics knobs:

- `lambda_sem`, `epsilon_match`
- `theta_carry` (Jaccard threshold)
- `delta_sem_max` (semantic guardrail)
- `topo_endpoint_eps` (endpoint movement guardrail)

See `witnessed_ph/config.py` for defaults.

---

## Reproducibility notes

To make future-you happy:

- Pin versions (Python, torch, transformers, spacy, gudhi).
- Record `diagram["library_versions"]` in any saved artifact.
- If you need *strict* reproducibility, pin the HuggingFace model revision (commit hash) and run with a local cache.

---

## Debugging checklist

If something refuses to run:

- **spaCy import errors:** ensure you have a spaCy/pydantic combination that works (this repo pins `pydantic<2`).
- **Model download failures:** confirm your HuggingFace cache + internet access.
- **Slow runs:** try `device="cuda"` and/or reduce token count via stricter `pos_filter` / stopwords.
- **No H1 bars:** lower `min_persistence` or increase `max_edge_length` (or accept that loops are rarer in short slices).

---

## Extending the repo

Common extensions that fit cleanly:

- Swap `embedding_model` (e.g. other transformer backbones).
- Add a different witness policy (e.g. choose witnesses from a minimal cycle basis for H1).
- Add visualisation (barcodes/diagrams; time-series of event counts).
- Replace greedy matching with a global assignment (Hungarian) if you want optimal matchings.

---

## License

MIT.
