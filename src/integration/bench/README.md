# Bench harness

`test_planner_bench.py` measures BCache `plan_window` latency/throughput
against the synthetic KV-tiering workload at three scales. CPU-only; the
module must finish in under 2 minutes.

```bash
make bench          # run benchmarks, print table, write out/bench_planner.json
pytest -m bench -s  # same, showing the per-case table
```
