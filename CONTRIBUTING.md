# Contributing

Issues and pull requests are welcome.

- **Measurements from other machines** are the most useful contribution: chip, unified memory, macOS version, decode and prompt tok/s from `bench/api_bench.py`, and whether the run was sustained (longer than ~10 s). Short runs are optimistic because the GPU holds its top clock only for a few seconds.
- **Engine changes** live in `patches/`, applied in order onto `danielhanchen/llama.cpp` at the commit pinned in `setup.sh`. Keep each patch focused, and check correctness with `test-backend-ops` against the CPU backend.
- **Speed claims** should come with an A/B that alternates the two variants in the same session, plus a quality check (identical greedy output or KL divergence) when weights or kernels change.

By contributing you agree that your contribution is licensed under the MIT License of this repository.
