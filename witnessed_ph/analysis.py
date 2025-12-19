"""witnessed_ph.analysis

High-level entry points:
- analyse_text_single_slice: compute a witnessed persistence diagram for one slice
- analyse_conversation: treat a list of turns as one slice and analyse
- helpers for printing and sorting bars

This is the "make it run" module intended for book reproduction and examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import re

import numpy as np

from .config import default_config, get_library_versions, resolve_device, set_global_seeds
from .embedding import (
    compute_pairwise_distances,
    load_embedding_model,
    load_spacy_model,
    segment_into_utterances,
    text_to_point_cloud,
)
from .h0 import compute_h0_theme_bars, compute_h0_raw_bars
from .h1 import compute_h1_bars_gudhi
from .utils import to_jsonable


def _maybe_strip_speaker(line: str) -> Tuple[Optional[str], str]:
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9_ \-]{0,32})\s*:\s*(.*)$", line)
    if not m:
        return None, line.strip()
    speaker = m.group(1).strip()
    content = m.group(2).strip()
    return speaker, content


def _segment_with_optional_speaker_stripping(text: str, mode: str, strip: bool) -> List[Tuple[str, Optional[str], str]]:
    if mode != "lines" or not strip:
        return segment_into_utterances(text, mode=mode)

    utterances: List[Tuple[str, Optional[str], str]] = []
    utt_idx = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        speaker, content = _maybe_strip_speaker(line)
        if content:
            utterances.append((f"u{utt_idx}", speaker, content))
            utt_idx += 1
    return utterances


def analyse_text_single_slice(
    text: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    segmentation_mode: str = "lines",
    verbose: bool = False,
    slice_id: str = "τ",
    embedding_model=None,
    embedding_tokenizer=None,
    nlp=None,
) -> Dict[str, Any]:
    """Analyse one text slice and return a witnessed persistence diagram.

    This function corresponds to the Chapter 4 single-slice pipeline:
    tokenise → contextual embeddings → distance matrix → persistence → witnesses.

    Parameters
    ----------
    text:
        Raw text of the slice.
    config:
        Overrides for `default_config()`.
    segmentation_mode:
        "lines" | "turns". ("sentences" is not implemented in the bundled embedding.py.)
    verbose:
        Print basic progress.
    slice_id:
        Stored in the output for provenance.
    embedding_model, embedding_tokenizer, nlp:
        Optional preloaded objects for efficiency in batch runs.

    Returns
    -------
    diagram dict with keys:
        slice_id, num_tokens, num_utterances, bars, config, library_versions, ...
    """
    cfg = default_config()
    if config:
        cfg.update(config)

    set_global_seeds(int(cfg.get("random_seed", 0)))

    # Resolve device for transformer model
    cfg["device"] = resolve_device(str(cfg.get("device", "auto")))

    strip = bool(cfg.get("strip_speaker_labels", True))
    seg_mode = segmentation_mode
    if segmentation_mode == "lines" and strip:
        # embedding.segment_into_utterances("turns") strips "Speaker:" prefixes safely
        seg_mode = "turns"
    utterances = segment_into_utterances(text, mode=seg_mode)

    if verbose:
        print(f"[witnessed_ph] slice {slice_id}: {len(utterances)} utterances")

    # Load spaCy + transformer model if not provided
    if nlp is None:
        nlp = load_spacy_model(str(cfg.get("spacy_model", "en_core_web_sm")))
    if embedding_model is None or embedding_tokenizer is None:
        if verbose:
            print(f"[witnessed_ph] loading embedding model: {cfg.get('embedding_model')}")
        embedding_model, embedding_tokenizer = load_embedding_model(str(cfg.get("embedding_model")))

    # Embed tokens → point cloud
    pc = text_to_point_cloud(
        text,
        config=cfg,
        segmentation_mode=seg_mode,
        nlp=nlp,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
    )

    embeddings = pc["embeddings"]
    token_ids = pc["token_ids"]
    tokens = pc["tokens"]

    # Distance matrix
    dist = compute_pairwise_distances(embeddings, metric=str(cfg.get("distance_metric", "cosine")))

    # H0 bars (either raw or Chapter-4 theme view)
    h0_mode = str(cfg.get("h0_mode", "theme")).lower()
    if h0_mode == "raw":
        h0_bars = compute_h0_raw_bars(
            embeddings=embeddings,
            token_ids=token_ids,
            tokens=tokens,
            distance_matrix=dist,
            max_edge_length=float(cfg.get("max_edge_length", 0.75)),
            min_persistence=float(cfg.get("min_persistence", 0.03)),
            min_witness_tokens=int(cfg.get("min_witness_tokens", 2)),
            max_witness_tokens=cfg.get("max_witness_tokens", 5),
            distance_metric=str(cfg.get("distance_metric", "cosine")),
        )
    else:
        h0_bars = compute_h0_theme_bars(
            embeddings=embeddings,
            token_ids=token_ids,
            tokens=tokens,
            distance_matrix=dist,
            max_edge_length=float(cfg.get("max_edge_length", 0.75)),
            min_persistence=float(cfg.get("min_persistence", 0.03)),
            min_witness_tokens=int(cfg.get("min_witness_tokens", 2)),
            max_witness_tokens=cfg.get("max_witness_tokens", 5),
            distance_metric=str(cfg.get("distance_metric", "cosine")),
        )

    # Optional H1 bars (requires gudhi)
    h1_bars: List[Dict[str, Any]] = []
    if int(cfg.get("max_dimension", 1)) >= 1:
        h1_bars = compute_h1_bars_gudhi(
            embeddings=embeddings,
            token_ids=token_ids,
            tokens=tokens,
            distance_matrix=dist,
            max_edge_length=float(cfg.get("max_edge_length", 0.75)),
            min_persistence=float(cfg.get("min_persistence", 0.03)),
            max_witness_tokens=cfg.get("max_witness_tokens", 5),
            distance_metric=str(cfg.get("distance_metric", "cosine")),
        )

    # Build output diagram
    diagram: Dict[str, Any] = {
        "slice_id": slice_id,
        "num_tokens": int(len(token_ids)),
        "num_utterances": int(len(pc.get("utterances", {}))),
        "config": cfg,
        "embedding_model": str(cfg.get("embedding_model")),
        "distance_metric": str(cfg.get("distance_metric")),
        "library_versions": get_library_versions(),
        "bars": [],
    }

    # Merge bars and add stable IDs
    bars: List[Dict[str, Any]] = []
    # (H0 first, then H1) and prefix ids by dimension for clarity
    for i, b in enumerate(h0_bars):
        bb = dict(b)
        bb["id"] = f"h0_{i}"
        bars.append(bb)
    for i, b in enumerate(h1_bars):
        bb = dict(b)
        bb["id"] = f"h1_{i}"
        bars.append(bb)

    # global sort by persistence
    bars.sort(key=lambda b: float(b.get("persistence", 0.0)), reverse=True)
    diagram["bars"] = bars

    if verbose:
        print_diagram_summary(diagram)

    return diagram


def analyse_conversation(
    turns: Sequence[Dict[str, str]],
    *,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Analyse a structured conversation as a *single* slice.

    `turns` is a sequence of dicts with keys {"speaker", "text"}.
    """
    # Render as Speaker: text lines
    lines: List[str] = []
    for t in turns:
        speaker = t.get("speaker", "").strip() or "Speaker"
        text = t.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    rendered = "\n".join(lines)

    return analyse_text_single_slice(
        rendered,
        config=config,
        segmentation_mode="lines",
        verbose=verbose,
        slice_id="conversation",
    )


def list_bars_by_persistence(
    diagram: Dict[str, Any],
    *,
    dim: Optional[int] = None,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return bars sorted by persistence (descending)."""
    bars = list(diagram.get("bars", []))
    if dim is not None:
        bars = [b for b in bars if int(b.get("dim", -1)) == int(dim)]
    bars.sort(key=lambda b: float(b.get("persistence", 0.0)), reverse=True)
    if top_n is not None:
        bars = bars[: int(top_n)]
    return bars


def print_diagram_summary(diagram: Dict[str, Any]) -> None:
    """Pretty-print a lightweight summary."""
    bars = list(diagram.get("bars", []))
    h0 = [b for b in bars if int(b.get("dim", -1)) == 0]
    h1 = [b for b in bars if int(b.get("dim", -1)) == 1]
    print(f"Diagram for slice {diagram.get('slice_id')}")
    print(f"  tokens:     {diagram.get('num_tokens')}")
    print(f"  utterances: {diagram.get('num_utterances')}")
    print(f"  bars:       {len(bars)}  (H0={len(h0)}, H1={len(h1)})")
    if bars:
        top = bars[0]
        print(
            f"  top bar:    {top.get('id')} dim={top.get('dim')} "
            f"[{top.get('birth'):.3f}, {top.get('death'):.3f}) pers={top.get('persistence'):.3f}"
        )
