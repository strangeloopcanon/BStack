from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]


def resolve(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return ROOT.joinpath(*parts)


from integration.data_pipeline import sample_feature_plan
from integration.kv_data_plane import build_cache_plan, simulate_cache_plan
from integration.weight_swapper import build_swap_plan, bucket_summary

try:
    from bstack_runtime.runtime import BwRuntime, WaveSpec
except Exception:  # pragma: no cover - optional bstack-runtime build
    BwRuntime = None  # type: ignore
    WaveSpec = None  # type: ignore


def prepare_demo_checkpoints(demo_root: Path) -> tuple[Path, Path]:
    prev_dir = demo_root / "hotweights_prev"
    next_dir = demo_root / "hotweights_next"
    prev_dir.mkdir(parents=True, exist_ok=True)
    next_dir.mkdir(parents=True, exist_ok=True)

    prev_file = prev_dir / "embedding.bin"
    if not prev_file.exists():
        prev_file.write_bytes(b"prev-embedding\n")
    next_file = next_dir / "embedding.bin"
    if not next_file.exists():
        next_file.write_bytes(b"next-embedding-v1\n")
    extra_file = next_dir / "layer1.bin"
    if not extra_file.exists():
        extra_file.write_bytes(b"layer1-stage\n")

    return prev_dir, next_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BStack demo pipeline")
    parser.add_argument("--output", type=Path, default=resolve("out"), help="Output directory for generated plans")
    parser.add_argument("--request-count", type=int, default=200, help="Synthetic requests to generate for the cache plan")
    parser.add_argument("--bucket-mb", type=int, default=32, help="Bucket size passed to hotweights planner")
    args = parser.parse_args(argv)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Generating cache plan via BCache ...")
    cache_result = build_cache_plan(request_count=args.request_count)
    cache_json = out_dir / "cache_plan.json"
    cache_result.plan.to_json(cache_json)
    metrics = simulate_cache_plan(cache_result)
    print(f"  ops={len(cache_result.plan.ops)} avg_finish_ms={metrics['avg_finish_ms']:.2f} prefetch={metrics['prefetch_timeliness']:.2f}")

    print("[2/3] Generating swap plan via hotweights ...")
    demo_root = resolve("src", "integration", "examples", "data")
    prev_dir, next_dir = prepare_demo_checkpoints(demo_root)
    swap_result = build_swap_plan(prev_dir, next_dir, bucket_mb=args.bucket_mb)
    swap_json = out_dir / "swap_plan.json"
    swap_result.plan.to_json(swap_json)
    buckets = bucket_summary(swap_result.buckets)
    print(f"  plan_id={swap_result.plan.plan_id} buckets={buckets}")

    print("[3/3] Sampling datajax plan ...")
    datajax_summary = sample_feature__plan()
    print("  stages=", datajax_summary["stages"])

    if BwRuntime is not None:
        try:
            print("[bonus] Attempting bstack-runtime submission (optional) ...")
            import array

            rt = BwRuntime()
            # Small 2x2 GEMM-style wave on CPU backend using host arrays
            spec = WaveSpec(bm=2, bn=2, bk=2, swap_begin=0, swap_end=3)
            A = array.array('f', [1, 2, 3, 4])
            B = array.array('f', [5, 6, 7, 8])
            C = array.array('f', [0, 0, 0, 0])
            a_ptr, _ = A.buffer_info()
            b_ptr, _ = B.buffer_info()
            c_ptr, _ = C.buffer_info()
            evt = rt.submit_wave(spec, a_ptr, b_ptr, c_ptr)
            rt.wait(evt, timeout_ms=0)
            print("  bstack-runtime submission succeeded; C=", list(C))
        except Exception as exc:  # pragma: no cover - depends on local build
            print(f"  bstack-runtime unavailable: {exc}")
    else:
        print("[bonus] bstack-runtime Python bindings not installed; skipping runtime probe")

    print(f"Plans written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())