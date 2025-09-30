# Beyond Monolithic Serving: The BStack for High-Performance, Composable AI

The world of large-scale AI is a battlefield against bottlenecks. Your expensive, powerful GPUs are often left starved, waiting for data. Rolling out a new model feels like a high-stakes, all-or-nothing gamble. Monolithic serving stacks, while powerful, can be rigid, opaque, and lock you into a single vendor's ecosystem.

What if we could break free from this? What if, instead of a monolithic mainframe, we had a set of specialized, interoperable microservices for our AI infrastructure?

This is the philosophy behind the **BStack**: a collection of four independent, open-source projects that form a composable, high-performance stack for modern AI. This repository demonstrates how they work together, but more importantly, it showcases each component as a powerful tool you can adopt individually.

## Context: Compiler Technology Meets AI Infrastructure

We built **Bodo**, a high-performance dataframe compiler that JIT-compiles pandas code into optimized parallel native execution. The BStack demonstrates how this compiler technology solves previously intractable AI infrastructure problems—complex planning logic (delta computation, KV-cache orchestration, trace replay) that was too slow in pure Python and too painful to maintain in C++.

The result: production-grade performance from readable, maintainable pandas code.

---

## The Four Pillars: Stars of the Show

The BStack is built on four specialized, standalone projects. Each one targets a specific, painful bottleneck in AI systems.

### 1. `hotweights`: For Zero-Downtime Model Upgrades
Repo: https://github.com/strangeloopcanon/hotweights

*   **The Problem:** Updating model weights in a live production environment is risky and often requires downtime. A typical "pull-and-replace" approach is slow, lacks verification, and makes rollbacks a nightmare.
*   **The Solution:** `hotweights` treats model swaps as a first-class operation. It computes an efficient delta between the old and new weights (using **Bodo JIT** to accelerate the delta computation and bucket packing), packages changed shards into memory-sized buckets, and orchestrates a verified swap across a cluster using auto-selected transports.
*   **What Makes It Useful:**
    *   **Bodo-Accelerated Delta Planning:** The manifest diff and bucket packing logic is JIT-compiled by Bodo from readable pandas code into fast native execution. This keeps planning sub-second even for manifests with 100k-1M shards, so the control plane never dominates end-to-end time.
    *   **Delta-Based Transfers:** Only moves changed weights (typically 1-10% for fine-tunes), dramatically reducing transfer time vs full model reload.
    *   **Multi-Transport with Auto-Selection:** CUDA-IPC for intra-node sharing (with adaptive windowing for backpressure), NCCL for inter-node leader broadcast, UCX/MPI fallbacks. Optional GPUDirect Storage (GDS) via KvikIO.
    *   **Verification & Observability:** Built-in hash verification, Prometheus metrics, structured plans (JSON), and coordinator-based orchestration with optional Redis HA backend.
    *   **Optimizer State Preservation (Training):** For same-shape weight updates, preserves Adam/AdamW momentum buffers with policy control (preserve/reset/attenuate). Shape mismatches currently reset to zeros—deeper transforms are future work.
*   **The Benefit:** **Safe, auditable rollouts with minimal downtime.** Deploy fine-tunes to inference clusters, synchronize weights in distributed training (when shapes match), and roll back with confidence. All operations produce verifiable plans and observable metrics.
*   **Status & Notes:** Delta planning (Bodo-accelerated) and multi-transport replication are production-ready. Optimizer state preservation works for same-shape updates; shape-changing transforms are stubs. KV cache migration offers conservative dtype/RoPE transforms (validate for your model). Multi-node perf validation and published benchmarks are planned.

### 2. `BCache`: The Intelligent Logistics Layer for Your KV-Cache
Repo: https://github.com/strangeloopcanon/BCache

*   **The Problem:** Your GPUs are burning cycles waiting for KV-cache data. Simple caching strategies like LRU/FIFO are blind to request patterns, leading to cache misses that stall the entire pipeline, especially in multi-tenant, long-context workloads.
*   **The Solution:** `BCache` is an out-of-band KV-cache planner. It analyzes request patterns, scores data for popularity and urgency, and creates an optimal plan to prefetch data before the GPU needs it. It turns thousands of small, random I/O operations into larger, coalesced transfers.
*   **What Makes It Useful:**
    *   **Bodo JIT-Compiled Planner:** The core planning logic—scoring requests by popularity and urgency, coalescing contiguous page intervals, applying bandwidth caps and tenant quotas—is written in readable pandas but JIT-compiled to native code by **Bodo**. This makes sub-20ms planning feasible even for thousands of concurrent requests. Without Bodo, this complexity would either be too slow (pure Python) or require rewriting in C++.
    *   **I/O Coalescing:** Identifies contiguous or overlapping page requests within each (node, tier, prefix_cluster, layer) group and merges them into large, efficient transfers that meet a minimum I/O threshold.
    *   **Multi-Tier Orchestration:** Coordinates movement across storage→CPU→GPU tiers with bandwidth caps, tenant quotas, and deadline-aware scheduling in a single optimization pass.
    *   **Pluggable Architecture:** Planner produces a protobuf-compatible `CachePlan` IR; executors can be engine-native (vLLM/SGLang adapters provided) or standalone (Python simulator, optional native CUDA/HIP/Level Zero backends).
    *   **Experimental Prefix Clustering:** Optional MinHash LSH clustering (A/B flag) for semantic similarity detection across prefixes. This is a research prototype for identifying requests that share KV structure even when prompts differ textually—not enabled by default.
*   **The Benefit:** **Better GPU utilization and more predictable latency.** The Bodo-compiled planner can make sophisticated I/O decisions fast enough to matter, and the modular design lets you integrate it with existing engines without rewriting executors.
*   **Status & Notes:** Planner (Bodo JIT) and Python simulator are production-ready. vLLM and SGLang adapters provide integration hooks. Native copy engines (CUDA/HIP) are optional builds. MinHash clustering is experimental (A/B flag). No end-to-end GPU benchmarks in this repo yet; focus is on planning and integration patterns.

### 3. `DataJAX`: "JAX for Data"
Repo: https://github.com/strangeloopcanon/datajax

*   **The Problem:** Feature engineering pipelines and offline analysis often involve messy, hard-to-reproduce pandas scripts. Scaling these workflows or translating them into performant, hardware-specific code is a significant challenge.
*   **The Solution:** `DataJAX` brings JAX-style functional transformations (`djit`, `vmap`, `pjit`, `scan`) to tabular data. It traces pandas-like operations into a compact Intermediate Representation (IR), which can then be lowered to different backends (pandas for fast iteration, **Bodo** for distributed execution).
*   **What Makes It Useful:**
    *   **JAX-Inspired API for DataFrames:** Write pandas code decorated with `@djit`, and DataJAX traces operations (filters, joins, aggregations) into an IR graph instead of executing immediately.
    *   **Reproducible Plans:** The IR can be serialized, inspected, and replayed deterministically—useful for turning messy production traces into clean, auditable execution plans.
    *   **Bodo Backend for Scale:** Lower the IR to Bodo's SPMD compiler to execute the same pandas-like code across an MPI cluster with predictable data partitioning. The `pjit` decorator lets you specify sharding (e.g., `shard.by_key("user_id")`) for distributed execution.
    *   **WaveSpec Generation:** DataJAX can export "wave specifications" (structured execution plans with timing, tile configs, I/O extents) that feed into `BCache` or other planners for offline tuning and replay.
*   **The Benefit:** **Reproducible, scalable data pipelines.** Trace messy pandas workflows into clean IR, replay them for offline analysis, or scale them across a cluster with Bodo—all from the same high-level code.
*   **Status & Notes:** Prototype. The trace→IR→plan→execute pipeline works with pandas and Bodo stub backends. Real Bodo lowering requires licensed Bodo installation and MPI environment. IR coverage includes basic filters, joins, and aggregations; UDF-heavy workloads and advanced window functions are future work.

### 4. `bw-runtime`: A Minimal, Swap-Aware Runtime
Repo: https://github.com/strangeloopcanon/bw-runtime

*   **The Problem:** Integrating custom planners and schedulers with existing deep learning engines is hard. You often need a thin, predictable layer to enforce scheduling decisions (like swap windows) and gather metrics without fighting the engine.
*   **The Solution:** `bw-runtime` provides a minimal, stable C ABI that acts as a clean contract between a planner and the underlying hardware. It's not another kernel library; it's a narrow runtime layer focused on swap-aware scheduling and planner feedback.
*   **What Makes It Useful:**
    *   **Narrow C ABI:** Engine-agnostic interface (`bwrt_submit_wave`, `bwrt_wait`, `bwrt_sample`) lets you plug it under any engine with minimal surface area.
    *   **Swap-Window Awareness:** Accepts `WaveSpec` plans with `swap_begin`/`swap_end` metadata, useful for coordinating with `hotweights` or other control planes.
    *   **Metrics Feedback:** Optional `bwrt_get_caps` and `bwrt_sample` provide runtime counters (e.g., active ops, completion stats) back to planners for closed-loop optimization.
    *   **Portable:** CPU fallback (async events + simple GEMM) works everywhere; CUDA stubs are optional build flags for future GPU-specific features.
*   **The Benefit:** **A stable seam for integration.** The narrow ABI demonstrates how to wire planner-produced schedules into a runtime without rewriting your engine.
*   **Status & Notes:** Current implementation is a **CPU fallback** for portability and integration testing. Command ring, sync primitives (barrier/mbarrier), and Python bindings (ctypes/pybind11) are implemented. CUDA stubs are optional. The roadmap targets Blackwell-era device features (persistent kernels, TMA, WGMMA), but the current repo focuses on demonstrating the ABI pattern and proving integration feasibility.

---

## The Bodo Advantage: Why It Matters

A key differentiator across `hotweights`, `BCache`, and `DataJAX` is how they leverage **Bodo** to accelerate planning and execution without sacrificing code readability.

**The Problem Space:** AI infrastructure planning—delta computation, bucket packing, KV-cache orchestration—is naturally expressed in dataframe operations. But these problems operate at scale (thousands of concurrent requests, 100k+ weight shards) where pure Python is too slow. The traditional answer was "rewrite in C++," which kills iteration speed and makes the codebase fragile.

**Applying Compiler Technology:** We built Bodo to JIT-compile pandas into optimized parallel native code. BStack demonstrates this technology solving real AI infrastructure bottlenecks. Write planning logic in readable pandas; Bodo compiles it to production-grade performance.

**Measured Impact:**
- `hotweights`: Delta planning completes sub-second for manifests with 100k-1M shards
- `BCache`: Planning decisions in sub-20ms even for thousands of concurrent KV requests  
- `DataJAX`: Same pandas code runs locally for iteration, then scales to MPI cluster for production

**Why This Matters:** AI infrastructure planning was previously a choice between "too slow" (Python) or "too rigid" (C++). Compiler technology breaks that tradeoff. The BStack shows how pandas-based planning can meet production SLOs when properly compiled.

---

## Shared Contracts (How They Click)

A tiny shared IR lets producers and consumers meet at stable seams:

- **CachePlan:** A windowed batch of typed data moves (TransferOps) with optional prefetch/evict hints.
- **SwapPlan:** The delta between two manifests plus TransferOps to realize it within a swap window.
- **TransferOp:** A typed move (H2D/D2H/P2P/STORAGE2H) with byte lengths, offsets, and optional KV page refs.
- **WaveSpec (bw-runtime):** Execution plan with tile shapes, I/O extents, timing constraints, and swap window metadata provided by the runtime layer.

These contracts keep planners and runtimes decoupled while ensuring compatibility.

## Proof in Action: Running the Stack

Talk is cheap. Let's see the components in action. The `integration/examples/run_stack.py` script in this repository runs a synthetic workload through the entire stack.

Example output from one run (values vary per run):

```text
[1/3] Generating cache plan via BCache ...
  ops=18 avg_finish_ms=9.31 prefetch=0.94

[2/3] Generating swap plan via hotweights ...
  plan_id=swap-next buckets=[{'bucket_id': 0, 'items': 2, 'bytes': 32}]

[3/3] Sampling datajax plan ...
  stages= ['input: InputStep', 'transform: MapStep', 'aggregate: AggregateStep']

[bonus] Attempting bw-runtime submission (optional) ...
  bw-runtime submission succeeded; C= [19.0, 22.0, 43.0, 50.0]
```

**What this tells us:**

1.  **BCache planned a window of coalesced transfers** with a high prefetch rate, ensuring data is ready ahead of time.
2.  **hotweights generated a `swap-next` plan**, producing a small delta (32 bytes in this demo) to update 2 items — the core of a zero-downtime update.
3.  **DataJAX traced the workload** into a clear, three-stage execution plan.
4.  **bw-runtime successfully received and executed** a small computational wave, proving the end-to-end integration.

This is the power of composition: four independent tools, each generating a precise plan, working in concert.

Note: This is a synthetic pipeline to illustrate composition and interfaces, not a performance benchmark.

---

## Architecture: A Layered, Composable Approach

If monolithic servers are mainframes, the BStack is the microservices revolution for AI infrastructure. The key is a layered design where each component has a distinct role, communicating through a tiny, stable API.

### Modern Architecture Diagram (Mermaid)

This diagram shows the recommended, engine-friendly posture. The control plane (BStack) orchestrates the data plane (your existing engine's executors) without rewriting kernels.

```mermaid
graph TD
    subgraph "User Space"
        Clients
    end

    subgraph "Control Plane (The BStack Differentiation)"
        A[DataJAX: Trace Analysis / WaveSpec Gen]
        B[BCache: Global KV-Cache Planner]
        C[hotweights: Versioning & Swap Planner]
    end

    subgraph "Integration Layer (Tiny IR)"
        D{Shared Plans: WaveSpec, CachePlan, SwapPlan}
    end

    subgraph "Data Plane (Leverage Existing Engines)"
        E[vLLM / SGLang / TRT-LLM Engine]
        F[Executors: HiCache / LMCache / KVBM]
        G[Transports: MPI / UCX / GDS]
    end
    
    subgraph "Device & Kernels"
        H[Your Existing Kernels]
        I[bw-runtime (Optional ABI for caps/metrics)]
    end

    Clients --> E
    A --> D
    B --> D
    C --> D
    D --> F
    D --> G
    E --> F
    F --> H
    G --> H
    F --> I
    
    style A fill:#cde4ff,stroke:#333,stroke-width:2px
    style B fill:#cde4ff,stroke:#333,stroke-width:2px
    style C fill:#cde4ff,stroke:#333,stroke-width:2px
```

### Original ASCII Diagram

The original diagram illustrates the flow of data and plans between the components.

```
Clients → Router → Engine(s) → Kernels
            │         │
            │         ├─ vLLM(+LMCache/KVBM) / SGLang(HiCache) / TRT‑LLM
            │         │     └ engine-local batching, attention, local KV
            │         │
            │   ┌─────▼──────────────────────────────┐
            │   │ Control plane                       │
            │   │ • BCache planner (global KV policy) │
            │   │ • hotweights ctl (versions, swaps)  │
            │   │ • datajax (WaveSpec, replay tuner)  │
            │   └─────┬──────────────┬───────────────┘
            │         │              │
            │   WaveSpec/prefetch    │ Swap plan (manifest+delta)
            │         │              │
            │   ┌─────▼──────────────▼────────────────────────┐
            │   │ Data plane (executors you don’t rewrite)    │
            │   │ • HiCache / LMCache / KVBM do actual moving │
            │   │ • Optional GDS/NVLink/NCCL where available  │
            │   │ • hotweights transport (MPI/UCX/IPC, GDS)   │
            │   └─────┬───────────────────────────────────────┘
            │         │ fences + ptrs + metrics
            │   ┌─────▼───────────────────────────────────────┐
            │   │ Device/runtime                              │
            │   │ • TRT‑LLM or engine kernels (default)       │
            │   │ • bw‑runtime optional (caps/metrics)        │
            │   └─────────────────────────────────────────────┘
            ▼
        Observability: Prometheus + WaveSpec replay
```

### Layer at a Glance

| Layer      | Module(s)                         | Primary role                                                   | Reuse from OSS                    | Competes with                     | Differentiator                           | Must‑ship work                                                                 |
| ---------- | --------------------------------- | -------------------------------------------------------------- | --------------------------------- | --------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| Control    | BCache + hotweights + DataJAX     | Global KV policy, versioned rollouts, trace→WaveSpec/replay    | Keep engine batching/attention    | Engine‑centric built‑in controllers | Cross‑engine KV logistics + safe rollouts | Adapters for prefetch windows; coordinator/canary/rollback; replay/tuning     |
| Data plane | Executors via connectors          | Do actual movement across tiers (RAM/SSD/GDS)                  | HiCache/LMCache/KVBM              | Rewriting executors               | Don’t rewrite; schedule globally          | Define/implement “prefetch window” APIs into executors; honor quotas/deadlines |
| Device     | bw‑runtime (optional)             | Caps/metrics ABI, swap‑window enforcement                      | TRT‑LLM kernels by default        | Custom DSL runtimes               | Portability + planner feedback           | Only if non‑NVIDIA or custom counters required right now                       |

Markdown tables render well on GitHub and most viewers; if yours doesn’t, the prose and diagrams above mirror the table.

---

## Use Them Independently

- `hotweights`: rolling upgrades/rollbacks for your existing engine; replace fragile reload scripts with verifiable swaps.
- `BCache`: evaluate coalescing, prefix clustering, and caps on your traces using the simulator.
- `DataJAX`: build reproducible trace→WaveSpec and feature pipelines on pandas or Bodo backends.
- `bw-runtime`: thin ABI for scheduling/counters when portability or precise device metrics are required.

---

## Try It Yourself in 5 Minutes

This repository makes it easy to see the BStack in action.

```bash
# Clone the repo with its submodules (the 4 projects)
git clone --recurse-submodules <this-repo-url>
cd <repo-dir>

# Set up the environment and install all components in editable mode
make bootstrap

# Run the end-to-end demo
python integration/examples/run_stack.py
```
You’ll see the live output from all four components, generating plans and executing a workload, with the final plans saved to the `integration/examples/out` directory for inspection.

If bw‑runtime’s shared library isn’t found, build it (CPU fallback):

```bash
cmake -S third_party/bw-runtime -B third_party/bw-runtime/build
cmake --build third_party/bw-runtime/build -j
ctest --test-dir third_party/bw-runtime/build --output-on-failure
```

Optional CUDA stubs:

```bash
cmake -S third_party/bw-runtime -B third_party/bw-runtime/build -DBWRT_ENABLE_CUDA=ON
cmake --build third_party/bw-runtime/build -j
```

## Non‑Goals and Scope

- Not a monolithic serving framework; this repo demonstrates seams and composable contracts.
- Defaults are CPU‑friendly; GPU fast paths are optional and hardware‑dependent.
- Does not replace engine kernels or executors; it orchestrates around them and leverages their strengths.

---

## Related Work & Integration Points

These components are designed to interoperate with, not replace, leading engines and executors:

- vLLM (PagedAttention, prefix caching): https://github.com/vllm-project/vllm
- SGLang HiCache (hierarchical/prefix caching): https://lmsys.org/blog/2025-09-10-sglang-hicache/
- NVIDIA TensorRT‑LLM (full‑stack runtime): https://github.com/NVIDIA/TensorRT-LLM
- LMCache (cross‑instance KV sharing): https://github.com/LMCache/LMCache
- NVIDIA KVBM (KV Bandwidth Manager): https://docs.nvidia.com/dynamo/latest/architecture/kvbm_architecture.html

In the recommended posture, engines keep their batching/attention kernels while the BStack provides global KV logistics and safe rollouts across engines.

---

## What's Production-Ready vs Research

**Production-Ready Today:**
- **hotweights delta planning and multi-transport:** Bodo-accelerated manifest diffing, bucket packing, and verified rollouts across CUDA-IPC/NCCL/UCX/MPI work now.
- **BCache Bodo-compiled planner:** The core scoring, coalescing, and multi-tier orchestration logic is fast enough for production use. vLLM/SGLang adapters provide integration hooks.
- **DataJAX trace→IR→plan pipeline:** Capturing pandas operations into reproducible IR and lowering to pandas or Bodo backends works for basic filters, joins, and aggregations.
- **Composable architecture:** The control/data plane separation with stable IR contracts (CachePlan, SwapPlan, WaveSpec) is a sound design pattern.

**Research/Experimental:**
- **BCache MinHash LSH clustering:** Semantic similarity detection for KV prefixes is implemented but experimental (A/B flag). Not benchmarked at scale.
- **hotweights optimizer state transforms:** Same-shape preservation works; shape-changing transforms are stubs.
- **hotweights KV cache migration:** Conservative dtype/RoPE transforms are opt-in; production validation for model families is ongoing.
- **bw-runtime GPU features:** Current implementation is CPU fallback; Blackwell-era persistent kernels and device metrics are roadmap items.

**Honest Take:**
These are **well-engineered, composable tools** that solve real problems (delta-based rollouts, fast planning, modular KV management). The Bodo-accelerated planning is a practical win—production-grade performance without C++ rewrites. Some advanced features are prototypes or future work, but the core value proposition (composable, observable, plan-driven infrastructure) is sound and usable today.

---

## Path to Production Validation (Appendix)

**To go from "useful tools" to "industry-proven":**

- **BCache:** Validate on real production traces from multi-tenant serving workloads. Publish benchmarks showing GPU utilization improvement and latency reduction vs baseline (LRU/FIFO). Harden native copy engines (CUDA/HIP) and prove end-to-end with actual vLLM/SGLang clusters.

- **hotweights:** Publish multi-node performance data (8-32 GPUs) showing delta transfer speedup vs full reload. Harden KV migration transforms for common model families (Llama, Mistral) with validation suites. Expand coordinator HA and safety checks for production rollout workflows.

- **DataJAX:** Broaden IR coverage (window functions, multi-key joins, UDF handling). Improve Bodo native lowering to eliminate Python UDFs and add native repartition. Publish comparative benchmarks vs pure pandas and hand-tuned Spark/Dask pipelines.

- **bw-runtime:** Move from CPU fallback to real device backends with persistent kernels. Surface actual runtime metrics (wgmma_active, tma_occ) from GPU to planners. Validate the ABI pattern with at least one production engine integration.

**The Gap:** Good engineering and promising architecture exist. What's missing are published benchmarks, large-scale validation, and proof that the integration patterns work in real production environments.

---

## The Bottom Line

The future of AI infrastructure is not monolithic; it's **composable**.

The BStack demonstrates how **compiler technology** can solve AI infrastructure bottlenecks that were previously stuck between "too slow" and "too complex." Instead of rewriting the world or locking into a monolithic stack, you can leverage specialized, focused tools built on compiled dataframe operations.

**What works today:**
- **hotweights** for delta-based, verified model rollouts with compiled delta planning
- **BCache** for fast, multi-tier KV-cache orchestration with compiled request scoring and coalescing
- **DataJAX** for reproducible, scalable DataFrame pipelines with compiled execution
- **bw-runtime** for demonstrating narrow ABI integration patterns

**The value proposition:**
- **Compiler-accelerated planning:** Write readable pandas code that meets production SLOs when compiled
- **Composable design:** Adopt components independently without rewriting engines
- **Novel application domain:** Applying dataframe compiler technology to LLM infrastructure planning
- **Observable operations:** Metrics, verification, and structured plans

We built Bodo to make high-performance dataframe processing accessible. BStack shows how that same compiler technology solves real AI infrastructure bottlenecks—model rollout friction, KV-cache I/O waste, messy data pipelines—with composable tools that work with existing infrastructure.

---

## External References

- SGLang HiCache: https://lmsys.org/blog/2025-09-10-sglang-hicache/
- NVIDIA KVBM: https://docs.nvidia.com/dynamo/latest/architecture/kvbm_architecture.html
- vLLM: https://github.com/vllm-project/vllm
- TensorRT‑LLM: https://github.com/NVIDIA/TensorRT-LLM
- LMCache: https://github.com/LMCache/LMCache
