"""witnessed_ph.temporal

Convenience wrappers for Chapter 4's *second* experimental block: bar dynamics.

Typical usage:

    diagrams = analyse_conversation_slices(turns)
    dynamics = track_bars_over_time(diagrams)

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .analysis import analyse_text_single_slice
from .config import default_config
from .embedding import load_embedding_model, load_spacy_model
from .tracking import track_bars_over_time


def analyse_conversation_slices(
    turns: Sequence[Dict[str, str]],
    *,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Analyse each turn as its own slice τ_i and return a list of diagrams."""
    cfg = default_config()
    if config:
        cfg.update(config)

    # Load heavy models once
    nlp = load_spacy_model(str(cfg.get("spacy_model", "en_core_web_sm")))
    model, tokenizer = load_embedding_model(str(cfg.get("embedding_model")))

    diagrams: List[Dict[str, Any]] = []
    for i, t in enumerate(turns):
        speaker = t.get("speaker", "").strip() or "Speaker"
        text = t.get("text", "").strip()
        if not text:
            continue
        rendered = f"{speaker}: {text}"
        diag = analyse_text_single_slice(
            rendered,
            config=cfg,
            segmentation_mode="lines",
            verbose=verbose,
            slice_id=f"τ{i}",
            embedding_model=model,
            embedding_tokenizer=tokenizer,
            nlp=nlp,
        )
        diagrams.append(diag)
    return diagrams


def analyse_conversation_dynamics(
    turns: Sequence[Dict[str, str]],
    *,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run full temporal pipeline: per-turn diagrams + H0 tracking."""
    diagrams = analyse_conversation_slices(turns, config=config, verbose=verbose)
    dynamics = track_bars_over_time(diagrams, dim=0, config=config)
    return {"diagrams": diagrams, "dynamics": dynamics}
