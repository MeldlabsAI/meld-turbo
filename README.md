<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" width="96" alt="Meld Labs">
  </picture>
</p>
<h1 align="center">Meld Turbo</h1>

<p align="center"><strong>A 125B-parameter model, fully local, on a MacBook.</strong><br>
Qwen3.8-Flash-Next at ~45 tokens/s on Apple silicon — no cloud, no API key, nothing leaves your Mac.</p>

<p align="center">🏁 <b>~45 tokens/s on a 2023 MacBook Pro (M2 Max) — and it fits in 64 GB</b><br>
<sub>64 GB measured in a simulation so far — <a href="https://github.com/MeldlabsAI/meld-turbo/issues/1">help us confirm it on a real one</a></sub></p>

<p align="center">
  <a href="https://meldlabs.ai/meld-turbo"><b>Project page</b></a> ·
  <a href="#quick-start"><b>Quick start</b></a> ·
  <a href="docs/PERFORMANCE.md">Performance notes</a>
</p>

<p align="center">
  <img alt="Apple silicon" src="https://img.shields.io/badge/Apple%20silicon-M2%20Max%2B-black">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-14%2B-black">
  <img alt="engine" src="https://img.shields.io/badge/engine-llama.cpp%20%2B%20patches-orange">
  <img alt="API" src="https://img.shields.io/badge/API-OpenAI--compatible-brightgreen">
  <img alt="MIT" src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

**Meld Turbo** runs **Qwen3.8-Flash-Next** — a Mixture-of-Experts model with 125B parameters, 6B of them active per token — on a single Mac, behind an OpenAI-compatible API. Any chat client, coding agent or script that speaks that API can use it.

Models of this size are usually served from a GPU cluster. A MoE model only reads a small part of its weights for each token, though, and Apple silicon puts a large, fast unified memory right next to the GPU. Meld Turbo is the set of engine changes, tools and settings that turns that into a usable local model: **~40–47 tokens/s** of generation on a MacBook Pro with an M2 Max and 96 GB.

It is the first open-source release from [Meld Labs](https://meldlabs.ai).

https://github.com/user-attachments/assets/5787a5b1-2af7-41f1-9bef-a13976c2e2f4

<p align="center"><sub>Real use on a MacBook Pro M2 Max, 96 GB: the built-in chat UI writing a Python script, thinking off. Idle pauses are trimmed; the speed is not edited.<br>
<b>No player above?</b> Open <a href="docs/media/meld-turbo-demo.mp4">docs/media/meld-turbo-demo.mp4</a> — the same video — in this repository.</sub></p>

## Results

MacBook Pro 14" · M2 Max (38-core GPU) · 96 GB · ~400 GB/s

| | |
|---|---|
| **Generation** | ~40–47 tok/s on code and English, ~41 tok/s on Chinese |
| **Prompt processing** | ~300 tok/s |
| **Context** | 128K tokens |
| **Memory** | ~45–48 GB, locked in RAM |
| **On disk** | ~68 GB (39 GB weights + 29 GB n-gram table) |

Numbers are end-to-end through the API with speculative decoding on. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for how they were measured.

## How it gets there

- **Speculative decoding with the model's own MTP head.** Three draft tokens per round, verified in one pass of the full model.
- **A smaller draft vocabulary.** The draft head scores 106K common tokens instead of all 248K. A draft step drops from 3.5 to 2.4 ms with no change in acceptance, including for Chinese.
- **Dense layers re-encoded to Q8_0.** On Apple GPUs, low-bit K-quant kernels are integer-ALU bound when they verify several tokens at once. Q8_0 decodes cheaply, so verification gets 14% faster and generation 19% faster. Re-encoding changes the model by a KL divergence of only 0.0026.
- **Metal kernels** for 2-bit experts and for the small-batch matrix–vector products that verification is made of.

## Quick start

You need a Mac with Apple silicon (Max-class GPU or better) and **64 GB of unified memory or more** (96 GB recommended), plus Xcode command-line tools, CMake, Python 3 with `numpy`, and about 110 GB of free disk.

Tested on 96 GB. **64 GB works in a simulation** on the 96 GB Mac (GPU memory capped at the 48 GB that macOS allows on a 64 GB machine, 32 GB of RAM locked away): with the default settings the server uses 44–47 GB and generates at ~44 tok/s, 36–42 tok/s sustained. Close memory-heavy apps first, and if the server does not start, try `CTX=32768`. Have a real 64 GB Mac? Please post a run in [issue #1](https://github.com/MeldlabsAI/meld-turbo/issues/1).

```zsh
git clone https://github.com/MeldlabsAI/meld-turbo && cd meld-turbo
./setup.sh                      # fetch llama.cpp, apply patches/, build with Metal

# models (pip install -U huggingface_hub)
M=~/.meld-turbo/models; mkdir -p $M
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "Q2_0/*.gguf" --local-dir $M
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" --local-dir $M
mv $M/MTP/*.gguf $M/ && mv $M/Q2_0/*.gguf $M/

# re-encode the dense layers to Q8_0 (~3 minutes); the n-gram shard is reused as is
python3 tools/requant_dense.py $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
                               $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-d8-00001-of-00002.gguf
ln -s Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-d8-00002-of-00002.gguf

./serve.sh                      # API at http://127.0.0.1:8080/v1, chat UI at http://127.0.0.1:8080
```

## Use it

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. Any API key and any model name will do.

```zsh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about unified memory."}]}'
```

**As a coding agent** — with [pi](https://github.com/earendil-works/pi-mono), add a provider to `~/.pi/agent/models.json`:

```json
"local-mac": {
  "baseUrl": "http://127.0.0.1:8080/v1", "api": "openai-completions", "apiKey": "local",
  "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false, "supportsStore": false, "thinkingFormat": "qwen"},
  "models": [{"id": "qwen3.8-flash-next", "name": "Qwen3.8-125B-MoE (local)", "reasoning": true, "contextWindow": 131072, "maxTokens": 8192}]
}
```

Then run `pi --provider local-mac --model qwen3.8-flash-next`. Add `--thinking off` for faster replies.

`serve.sh` settings: `PORT`, `CTX` (context, default 131072), `NP` (parallel slots, default 1), `NPREDICT` (reply cap, default 16384), `LISTEN=0.0.0.0` to serve your LAN. Thinking is on by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`.

## What's in this repository

| Path | |
|---|---|
| `patches/` | 12 patches on llama.cpp (`danielhanchen/llama.cpp`, branch `qwen4exp/mtp`, pinned in `setup.sh`) |
| `setup.sh`, `serve.sh` | build and run |
| `tools/requant_dense.py` | re-encode dense K-quant / IQ4_XS tensors to Q8_0, copying experts byte for byte |
| `tools/gpu_pstate.c` | GPU frequency and performance-state logger (no root needed) |
| `bench/api_bench.py` | generation and prompt speed through the API |
| `data/draft_vocab.bin` | the 106,299-token draft vocabulary |
| `docs/PERFORMANCE.md` | measurements: what helped, what did not, known limits |

## Known limits

- **Sustained load.** The GPU holds its top clock for a few seconds, then settles about 20% lower. Long sessions run slower than short benchmarks.
- **Quality.** The experts are 2-bit. Quality against the original BF16 model has not been evaluated.
- **First turn of long sessions.** At ~300 tok/s of prompt processing, a long first prompt takes a while. Later turns reuse the prompt cache.
- **One request at a time** by default.

Results from other Macs are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## About Meld Labs

[Meld Labs](https://meldlabs.ai) builds AI infrastructure — inference, training and research compute, agent sandboxes — so that developers can spend their time on ideas instead of plumbing. Meld Turbo is our first open-source project: a community release for running a frontier-scale model on your own machine.

## Credits and license

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and Unsloth's MTP branch of it. The model is by the Qwen team, the GSQ-RCO quantizations are by ISTA-DASLab, the MTP head GGUF is by Unsloth, and the draft vocabulary comes from [Strata](https://github.com/Niko1221/Strata) (MIT). Third-party notices are in [NOTICE](NOTICE).

MIT — © 2026 The Meld Labs Authors. Model weights are under their own licenses.
