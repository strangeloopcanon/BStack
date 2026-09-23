"""Weight-swap x KV-cache demo: selective invalidation beats full flush."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bstack.paths import add_third_party_to_path, resolve

add_third_party_to_path()

from integration.swap_analysis import (
    analyze_swap_kv,
    make_checkpoints,
    report_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the swap x KV invalidation demo")
    parser.add_argument("--output", type=Path, default=resolve("out"))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument(
        "--changed",
        type=int,
        nargs="*",
        default=[2, 5],
        help="Layer indices re-trained in v1",
    )
    parser.add_argument(
        "--kv-pages", type=int, default=64, help="KV pages per layer resident in GPU"
    )
    args = parser.parse_args(argv)

    workdir = args.output / "swap_kv"
    workdir.mkdir(parents=True, exist_ok=True)
    print("Building synthetic checkpoints ...")
    v0, v1 = make_checkpoints(
        workdir, n_layers=args.layers, changed_layers=tuple(args.changed)
    )

    print("Planning weight swap + KV invalidation ...")
    result = analyze_swap_kv(
        v0, v1, n_layers=args.layers, kv_pages_per_layer=args.kv_pages
    )
    rep = report_dict(result)

    print(f"  changed layers:      {rep['changed_layers']}")
    print(f"  changed tensors:     {len(rep['changed_tensors'])}")
    print(f"  swap transfer:       {rep['swap_bytes'] / 1e6:.2f} MB")
    print(f"  KV cache total:      {rep['kv_total_mb']:.2f} MB")
    print(f"  KV invalidated:      {rep['kv_invalidated_mb']:.2f} MB")
    print(
        f"  KV reusable:         {rep['kv_reusable_mb']:.2f} MB "
        f"({rep['reuse_fraction']:.0%})"
    )
    print(f"  saved vs full flush: {rep['saved_vs_full_flush_mb']:.2f} MB")

    (args.output / "swap_kv.json").write_text(json.dumps(rep, indent=2))
    result.plan.to_json(args.output / "swap_kv_plan.json")
    print(
        f"\nWrote {args.output / 'swap_kv.json'} and {args.output / 'swap_kv_plan.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
