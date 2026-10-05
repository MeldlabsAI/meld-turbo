#!/usr/bin/env python3
"""Decode / prompt speed through an OpenAI-compatible /v1/chat/completions endpoint (streaming).

Uses a code-review style prompt built from a source tree, greedy (temperature 0), fixed max_tokens,
thinking off where the server honours it. Reports time to first token, decode tok/s (from the first
to the last streamed token) and prompt tok/s (prompt tokens / TTFT).

    python3 bench/api_bench.py --url http://127.0.0.1:8080 --src ./some/source/tree --ctx 1000,4000 -n 256 -r 3
"""
import argparse
import json
import os
import time
import urllib.request


def build_prompt(src, approx_tokens):
    exts = (".cpp", ".hpp", ".cu", ".py", ".h")
    files = []
    for root, _, names in os.walk(os.path.expanduser(src)):
        for n in sorted(names):
            if n.endswith(exts):
                files.append(os.path.join(root, n))
    files.sort()
    budget, parts = approx_tokens * 3.5, []  # ~3.5 chars per token for code
    for p in files:
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        parts.append(f"### {os.path.relpath(p, os.path.expanduser(src))}\n{text}\n")
        if sum(map(len, parts)) >= budget:
            break
    code = "".join(parts)[: int(budget)]
    return ("Review the following source files. List the three most likely bugs, each with the file, "
            "the line and a one-paragraph explanation.\n\n" + code)


def run(url, prompt, n, model):
    body = json.dumps({
        "model": model, "stream": True, "temperature": 0, "max_tokens": n,
        "messages": [{"role": "user", "content": prompt}],
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
    gen = usage.get("completion_tokens") or chunks
    pt = usage.get("prompt_tokens", 0)
    ttft = (first or time.perf_counter()) - t0
    dec = (gen - 1) / (last - first) if first and last and last > first and gen > 1 else 0.0
    return {"prompt_tokens": pt, "gen_tokens": gen, "ttft_s": round(ttft, 3),
            "prompt_tps": round(pt / ttft, 1) if ttft else 0, "decode_tps": round(dec, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--src", required=True, help="source tree the prompt is built from")
    ap.add_argument("--ctx", default="1000,4000", help="approximate prompt lengths in tokens")
    ap.add_argument("-n", type=int, default=256, help="max generated tokens")
    ap.add_argument("-r", type=int, default=3, help="runs per length")
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for c in [int(x) for x in a.ctx.split(",")]:
        prompt = build_prompt(a.src, c)
        res = []
        for i in range(a.r):
            # a different first line per run so no server-side prompt cache is reused
            res.append(run(a.url, f"[run {i} {time.time()}]\n" + prompt, a.n, a.model))
            print(json.dumps({"label": a.label, "ctx": c, "run": i, **res[-1]}), flush=True)
        dec = sorted(r["decode_tps"] for r in res)
        pp = sorted(r["prompt_tps"] for r in res)
        print(f"# {a.label} ~{c} tok prompt ({res[0]['prompt_tokens']}): decode median {dec[len(dec) // 2]} tok/s, "
              f"prompt median {pp[len(pp) // 2]} tok/s, ttft {sorted(r['ttft_s'] for r in res)[len(res) // 2]} s",
              flush=True)


if __name__ == "__main__":
    main()
