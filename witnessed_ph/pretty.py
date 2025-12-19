"""Pretty-print helpers for Chapter 4 (CLI / 'theme score' output).

The book's Chapter 4 uses a deliberately "hacker-ish" textual rendering of the
Step–Witness Log (SWL).  This module keeps that presentation logic out of the
core algorithms so it stays easy to extend.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple


_EVENT_SYMBOL = {
    "birth": ("•", "spawn"),
    "carry": ("→", "carry"),
    "drift": ("~", "drift"),
    "rupture": ("×", "rupture"),
    "rupture_semantic": ("×", "rupture"),
    "reentry": ("F", "re-entry"),
    "reentry_semantic": ("F", "re-entry"),
}


def _theme_label_from_bar(bar: Mapping[str, Any]) -> str:
    """Heuristic label for a theme bar (matches book-style captions).

    We use the first witness lemma if available; otherwise fall back to the bar id.
    """
    witness = bar.get("witness") or {}
    toks = (witness.get("tokens") or {})
    lemmas = toks.get("lemma") or []
    surfaces = toks.get("surface") or []
    lemma_set = witness.get("lemma_set") or []

    for cand in (lemmas, surfaces, lemma_set):
        if cand:
            try:
                return str(cand[0]).upper()
            except Exception:
                pass

    return str(bar.get("id", "THEME")).upper()


def _witness_snippet(bar: Mapping[str, Any], max_tokens: int = 6) -> str:
    witness = bar.get("witness") or {}
    toks = (witness.get("tokens") or {})
    surfaces = toks.get("surface") or []
    if not surfaces:
        lemma_set = witness.get("lemma_set") or []
        surfaces = [str(x) for x in lemma_set]
    if not surfaces:
        return "{∅}"
    snippet = surfaces[: max(1, max_tokens)]
    return "{" + ", ".join(str(x) for x in snippet) + "}"


def print_transition_summary(transitions: Iterable[Mapping[str, Any]]) -> None:
    """Print the per-step transition counts (birth/carry/drift/rupture/re-entry)."""
    rows = list(transitions)
    if not rows:
        print("(no transitions)")
        return

    header = [
        "step",
        "birth",
        "carry",
        "drift",
        "rupture",
        "re-entry",
    ]

    print("\nTRANSITION SUMMARY")
    print("-" * 72)
    print(
        f"{header[0]:>6}  {header[1]:>5}  {header[2]:>5}  {header[3]:>5}  {header[4]:>7}  {header[5]:>7}"
    )
    for t in rows:
        step = f"τ{t.get('t', '?')}→τ{t.get('t', '?') + 1 if isinstance(t.get('t'), int) else '?'}"
        print(
            f"{step:>6}  {int(t.get('birth', 0)):>5}  {int(t.get('carry', 0)):>5}  {int(t.get('drift', 0)):>5}  {int(t.get('rupture', 0)):>7}  {int(t.get('reentry', 0)):>7}"
        )
    print("-" * 72)


def print_theme_score(
    diagrams: List[Mapping[str, Any]],
    dynamics: Mapping[str, Any],
    *,
    max_tokens: int = 6,
    max_lines_per_slice: int = 0,
    dim: int = 0,
) -> None:
    """Print a "theme score" style rendering of the SWL.

    Parameters
    ----------
    diagrams:
        The list of per-slice diagrams returned by `analyse_conversation_slices`.
    dynamics:
        The dynamics dict returned by `track_bars_over_time` (via
        `analyse_conversation_dynamics`).
    max_tokens:
        Maximum witness tokens to display per theme.
    max_lines_per_slice:
        If >0, cap the number of theme lines printed per slice (useful for long
        conversations). 0 means "no cap".
    dim:
        Which homology dimension to display (Chapter 4 theme score is H0 => dim=0).
    """

    journeys = list(dynamics.get("journeys", []) or [])

    # Build lookup for bars per slice.
    bars_by_slice: List[Dict[str, Mapping[str, Any]]] = []
    for d in diagrams:
        bars = [b for b in (d.get("bars") or []) if int(b.get("dim", -1)) == int(dim)]
        bars_by_slice.append({str(b.get("id")): b for b in bars})

    # Index events by slice time.
    events_by_t: Dict[int, List[Tuple[str, Mapping[str, Any]]]] = {}
    for j in journeys:
        jid = str(j.get("id"))
        for ev in (j.get("events") or []):
            t = ev.get("t")
            if isinstance(t, int):
                events_by_t.setdefault(t, []).append((jid, ev))

    print("\nTHEME SCORE (SWL)")
    print("Legend: • spawn  → carry  ~ drift  × rupture  F re-entry")
    print("=" * 72)

    for t in range(len(diagrams)):
        slice_id = diagrams[t].get("slice_id", f"τ{t}")
        print(f"\n{slice_id}")

        # Build rows: (rank, line)
        rows: List[Tuple[float, str]] = []

        for jid, ev in events_by_t.get(t, []):
            ev_type = str(ev.get("type"))
            sym, label = _EVENT_SYMBOL.get(ev_type, ("?", ev_type))

            bar = None
            bar_id = ev.get("bar_id")
            if bar_id is not None:
                bar = bars_by_slice[t].get(str(bar_id))

            # For ruptures, try to show the last known bar (from previous slice)
            if bar is None and ev_type.startswith("rupture") and t > 0:
                prev_id = ev.get("from")
                if prev_id is not None:
                    bar = bars_by_slice[t - 1].get(str(prev_id))

            # Score for sorting: persistence if available.
            pers = 0.0
            if bar is not None:
                try:
                    pers = float(bar.get("persistence", 0.0))
                except Exception:
                    pers = 0.0

            if bar is not None:
                theme = _theme_label_from_bar(bar)
                w = _witness_snippet(bar, max_tokens=max_tokens)
                extra = ""
                if ev_type in {"carry", "drift"}:
                    jacc = ev.get("jaccard")
                    dbar = ev.get("d_bar")
                    parts = []
                    if isinstance(jacc, (int, float)):
                        parts.append(f"J={jacc:.2f}")
                    if isinstance(dbar, (int, float)):
                        parts.append(f"Δbar={dbar:.3f}")
                    if parts:
                        extra = "  " + "  ".join(parts)

                line = f"  {sym} [{jid}] {theme:<16} pers={pers:.3f}{extra}  w={w}"
            else:
                line = f"  {sym} [{jid}] (no bar data)"

            rows.append((pers, line))

        rows.sort(key=lambda x: x[0], reverse=True)
        if max_lines_per_slice and len(rows) > max_lines_per_slice:
            rows = rows[:max_lines_per_slice]
            rows.append((0.0, "  … (truncated; adjust max_lines_per_slice to show more)"))

        if not rows:
            print("  (no tracked themes)")
        else:
            for _, line in rows:
                print(line)
