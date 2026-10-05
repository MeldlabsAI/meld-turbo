# Notes for contributors and coding agents

- The engine is llama.cpp plus the patches in `patches/`; `setup.sh` pins the base commit. Do not vendor llama.cpp into this repository.
- Environment switches added by the patches use the `MELD_` prefix (`MELD_DRAFT_VOCAB`, `MELD_HC_MIX`, `MELD_BENCH_*`); debug ones are off by default.
- Every speed change needs a correctness check: `test-backend-ops` for kernels, identical greedy output or KL divergence for model-level changes.
- Measure sustained, not peak: alternate A/B runs and keep each run longer than ~10 s.
- English only in code, comments and docs.
