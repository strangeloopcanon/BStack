"""Tests for the weight-swap x KV-cache invalidation analysis."""

from __future__ import annotations

from integration.swap_analysis import (
    analyze_swap_kv,
    layer_of_tensor,
    make_checkpoints,
    report_dict,
)


def test_layer_of_tensor() -> None:
    assert layer_of_tensor("layer3.qkv.npy") == 3
    assert layer_of_tensor("layer12.mlp.npy") == 12
    assert layer_of_tensor("embed.bin") is None


def test_selective_invalidation(tmp_path) -> None:
    v0, v1 = make_checkpoints(tmp_path, n_layers=8, changed_layers=(2, 5), seed=11)
    res = analyze_swap_kv(v0, v1, n_layers=8, kv_pages_per_layer=16)

    # Only the two re-trained layers changed.
    assert res.changed_layers == [2, 5]
    assert len(res.changed_tensors) == 4  # qkv + mlp per changed layer
    assert res.swap_bytes > 0

    # KV pages: changed layers invalidated, the rest reusable.
    assert res.kv_invalidated_pages == 2 * 16
    assert res.kv_reusable_pages == 6 * 16
    evict_layers = {ref.layer for ref in res.plan.evict}
    assert evict_layers == {2, 5}
    assert len(res.plan.evict) == 2 * 16

    rep = report_dict(res)
    assert rep["reuse_fraction"] == 0.75
    assert rep["saved_vs_full_flush_mb"] > 0


def test_no_change_no_invalidation(tmp_path) -> None:
    v0, v1 = make_checkpoints(tmp_path, n_layers=4, changed_layers=(), seed=3)
    res = analyze_swap_kv(v0, v1, n_layers=4, kv_pages_per_layer=8)
    assert res.changed_layers == []
    assert res.swap_bytes == 0
    assert res.kv_invalidated_pages == 0
    assert res.kv_reusable_pages == 4 * 8
    assert report_dict(res)["reuse_fraction"] == 1.0


def test_full_retrain_invalidates_everything(tmp_path) -> None:
    changed = tuple(range(4))
    v0, v1 = make_checkpoints(tmp_path, n_layers=4, changed_layers=changed, seed=5)
    res = analyze_swap_kv(v0, v1, n_layers=4, kv_pages_per_layer=8)
    assert res.changed_layers == [0, 1, 2, 3]
    assert res.kv_reusable_pages == 0
    assert report_dict(res)["reuse_fraction"] == 0.0
