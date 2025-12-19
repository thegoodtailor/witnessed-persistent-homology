import numpy as np

from witnessed_ph.h0 import compute_h0_theme_bars, compute_h0_raw_bars
from witnessed_ph.tracking import track_bars_over_time


def _make_pc(n=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # normalise to unit sphere
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    token_ids = [f"t{i}" for i in range(n)]
    tokens = {tid: {"id": tid, "text": tid, "lemma": tid, "pos": "NOUN", "utterance_id": "u0", "char_start": 0, "char_end": 1, "embedding": X[i]} for i, tid in enumerate(token_ids)}
    # simple angular distances
    sim = X @ X.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = (np.arccos(sim) / np.pi).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return X, token_ids, tokens, dist


def test_h0_theme_births_can_be_positive():
    X, token_ids, tokens, dist = _make_pc()
    bars = compute_h0_theme_bars(
        embeddings=X,
        token_ids=token_ids,
        tokens=tokens,
        distance_matrix=dist,
        max_edge_length=1.0,
        min_persistence=0.0,
        min_witness_tokens=2,
        max_witness_tokens=5,
        distance_metric="cosine",
    )
    assert bars, "expected some theme bars"
    assert any(b["birth"] > 0 for b in bars), "theme bars should often have birth > 0"


def test_h0_raw_births_are_zero():
    X, token_ids, tokens, dist = _make_pc()
    bars = compute_h0_raw_bars(
        embeddings=X,
        token_ids=token_ids,
        tokens=tokens,
        distance_matrix=dist,
        max_edge_length=1.0,
        min_persistence=0.0,
        min_witness_tokens=1,
        max_witness_tokens=5,
        distance_metric="cosine",
    )
    assert bars
    assert all(abs(float(b["birth"])) < 1e-9 for b in bars)


def test_tracking_basic():
    # two slices with a single obvious carry
    bar0 = {"id": "h0_0", "dim": 0, "birth": 0.1, "death": 0.5, "witness": {"centroid": [1.0, 0.0], "lemma_set": ["climate", "change"]}}
    bar1 = {"id": "h0_0", "dim": 0, "birth": 0.12, "death": 0.52, "witness": {"centroid": [0.99, 0.01], "lemma_set": ["climate", "change"]}}

    d0 = {"slice_id": "τ0", "bars": [bar0], "config": {"lambda_sem": 0.5, "epsilon_match": 0.8, "theta_carry": 0.4, "delta_sem_max": 0.6, "topo_endpoint_eps": 0.2}}
    d1 = {"slice_id": "τ1", "bars": [bar1], "config": d0["config"]}

    res = track_bars_over_time([d0, d1], dim=0)
    assert res["transitions"][0]["carry"] == 1
