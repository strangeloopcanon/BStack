# bw-stack integration repo

This repository composes:

- [hotweights](https://github.com/strangeloopcanon/hotweights)
- [BCache](https://github.com/strangeloopcanon/BCache)
- [datajax](https://github.com/strangeloopcanon/datajax)
- [bw-runtime](https://github.com/strangeloopcanon/bw-runtime)

The goal is to provide a thin integration layer with shared plan APIs, demo runners, and reproducible tooling while keeping each component in its upstream repository.

## Highlights

- Pinned submodules under `third_party/` with lock verification (`stack.lock`, `make sync`).
- Shared plan IR defined in `src/bw_apis/` (protobuf schema + Python/C++ helpers).
- Synthetic end-to-end demo combining BCache cache plans, hotweights swap plans, DataJAX analytics, and optional bw-runtime probes.
- Simple Makefile for bootstrapping, codegen, linting, tests, and demo execution.

See `docs/README.md` for full usage details.
