# Performance notes

All numbers: MacBook Pro 14" (M2 Max, 38-core GPU, 96 GB, ~400 GB/s), Qwen3.8-Flash-Next GSQ-RCO Q2_0, llama.cpp Metal backend.

## Where the time goes

A 4-token MTP verify pass reads ~6–7.7 GB: dense projections (~3.4 GB as Q8_0), the 40 distinct experts the 4 tokens select (~2.65 GB of Q2_0), and hyper-connection weights (~1.25 GB of BF16). Consecutive tokens never shared an expert in our routing dumps (0 of 1,927 layer×window samples), so grouping experts across verify tokens does not help.

## Dense K-quants are integer-ALU bound at small batches

Effective bandwidth of an 8192×2560 mat-vec (weight bytes / time):

| Format | 1 token | 4 tokens |
|---|---:|---:|
| Q3_K | 133 GB/s | 57 GB/s |
| IQ4_XS | 226 | 73 |
| Q4_K | 216 | 72 |
| Q5_K | 170 | 85 |
| Q6_K | 253 | 111 |
| Q4_0 | 302 | 112 |
| **Q8_0** | **384** | **216** |
| BF16 | — | 302 |

The GSQ-RCO Q2_0 file keeps ~1.8 GB of dense projections in K-quant / IQ4_XS formats. `tools/requant_dense.py` re-encodes them to Q8_0 (1.77 → 3.46 GB; experts copied byte for byte):

| | Original | Re-encoded (d8) |
|---|---:|---:|
| 4-token verify pass | 53.9 ms | 46.5 ms (−13.8%) |
| 1-token pass | 30.5 ms | 32.5 ms (+6.6%) |
| Server + MTP decode | ~31 tok/s | ~37 tok/s (+19%) |
| KLD vs original (30×512 tokens) | — | mean 0.0026, top-1 agreement 98.5%, PPL ratio 1.0015 ± 0.0013 |
| Greedy 64 tokens | — | identical text |

## MTP draft vocabulary subset

`MELD_DRAFT_VOCAB=<file of int32 token ids>` makes the draft head score only those rows of the LM head (copied once into a small head), when backend sampling picks every draft output. The default list (106,299 tokens) covers English, code and CJK.

| Draft vocabulary | Draft step | Acceptance | Decode (Chinese summary task) |
|---|---:|---|---:|
| Full (248,320) | 3.51 ms | 0.66 / 0.59 | 35.4 tok/s |
| 106K subset | 2.38 ms | 0.66 / 0.64 | 41.0 tok/s |
| 40K English-only subset | 1.91 ms | 0.18 | 20.0 tok/s |

English-only subsets collapse on Chinese text. Use the 106K list for general use.

## Sustained load

The GPU holds its top frequency (1398 MHz) for only ~3 s of continuous load; then a 4-token verify pass goes from ~48 ms to ~63–70 ms (measured with `tools/gpu_pstate.c` and Metal command-buffer timestamps). Thermal state stays "Nominal", so this looks like a power budget. Short benchmarks are therefore optimistic. Judge changes by sustained runs or end-to-end server throughput, and alternate A/B runs.

## 64 GB Macs (simulated)

Simulated on the 96 GB MacBook Pro: `sudo sysctl iogpu.wired_limit_mb=48000` (the GPU limit macOS uses on a 64 GB machine) and a helper process holding 32 GiB of RAM locked. `bench/api_bench.py`, ~1,000-token code prompt unless noted:

| Model, context | Server memory | Decode | Prompt |
|---|---:|---:|---:|
| Original Q2_0, 16K | 39–40 GB | 33.3 tok/s | 259 tok/s |
| Q2_0-d8, 16K | 41–42 GB | 38.4 tok/s | 254 tok/s |
| Q2_0-d8, 32K | 41–42 GB | 35.7 tok/s | 234 tok/s |
| Q2_0-d8, 128K (default) | 44–45 GB | 35.3 tok/s | 222 tok/s |
| Q2_0-d8, 128K, ~8,000-token prompt | 46 GB | 31.4 tok/s | 205 tok/s |

All runs started and finished normally. Most layers are linear-attention, so the context costs little memory. The rest of the system is squeezed (~5% free, other apps compressed or swapped), so close memory-heavy apps on a 64 GB Mac.

## Tried, no gain

- Fused hyper-connection op (`MELD_HC_MIX=1`): correct, 31/31 op tests, identical output, but ±3% end to end.
- Faster q8_0 small-batch kernels (activations staged in threadgroup memory; multi-column n=1 kernel): no better than the existing kernel on tall matrices.
- Simdgroup-matrix kernels for 2–8 tokens: 1.7–2× slower.
- Deferring the draft catch-up decode into the first draft step: correct, no end-to-end gain.
- Draft head on the CPU; separate Metal command queues; raising thread QoS; GPU keep-alive kernels: no gain.
- `--spec-draft-n-max` 3 vs 4: equal within noise.
- `-bs` (GPU sampling): greedy output and acceptance identical with or without it.
