#!/usr/bin/env python3
"""Rewrite a GGUF with its dense K-quant / IQ4_XS projections converted to Q8_0.

Why: on Apple GPUs the K-quant mat-vec kernels are integer-ALU bound. At the 4-token batch of an
MTP verify pass, Q3_K / Q4_K / IQ4_XS run at only 57-73 GB/s and Q5_K / Q6_K at 85-111 GB/s,
against 216 GB/s for Q8_0 (M2 Max, 8192x2560). Q8_0 is ~2x the bytes but much
cheaper to decode, and re-encoding already-quantized values in Q8_0 is almost lossless.

Experts, token embedding (read one row per token), BF16/F32 and Q2_0/Q4_0 tensors are copied
byte for byte. Metadata, split keys and tensor order are kept, so a split model still loads with
its other shards (symlink them next to the output).

    python3 tools/requant_dense.py <in.gguf> <out.gguf> [--types Q3_K,Q4_K,...] [--keep output.weight] [--only REGEX] [--dry-run]

--only limits the conversion to tensors whose name matches REGEX (e.g. --types BF16 --only 'hc_.*_down').
"""
import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("MELD_HOME", os.path.expanduser("~/.meld-turbo")), "llama.cpp", "gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402

DEFAULT_TYPES = "Q3_K,Q4_K,Q5_K,Q6_K,IQ4_XS"


def wanted(t, types):
    return (QT(t.tensor_type) in types and "_exps" not in t.name and t.name != "token_embd.weight"
            and len(t.shape) >= 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--types", default=DEFAULT_TYPES)
    ap.add_argument("--keep", default="", help="comma-separated tensor names to copy unchanged")
    ap.add_argument("--only", default="", help="regex; convert only tensors whose name matches")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    types = {QT[x] for x in a.types.split(",")}

    r = gguf.GGUFReader(a.src)
    keep = set(filter(None, a.keep.split(",")))
    only = re.compile(a.only) if a.only else None
    conv = [t for t in r.tensors if wanted(t, types) and t.name not in keep and (only is None or only.search(t.name))]
    before = sum(int(t.n_bytes) for t in conv)
    after = sum(int(t.n_elements) // 32 * 34 for t in conv)
    print(f"{len(conv)} tensors -> Q8_0: {before / 1e9:.3f} GB -> {after / 1e9:.3f} GB")
    if a.dry_run:
        return
    if os.path.exists(a.dst):
        sys.exit(f"{a.dst} exists")

    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(a.dst + ".part", arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)

    names = {t.name for t in conv}
    for t in r.tensors:
        if t.name in names:
            shape = gguf.quant_shape_from_byte_shape(t.data.shape, QT(t.tensor_type))
            bshape = gguf.quant_shape_to_byte_shape(shape, QT.Q8_0)
            w.add_tensor_info(t.name, bshape, np.dtype(np.uint8), int(np.prod(bshape)), QT.Q8_0)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    t0, worst = time.time(), 0.0
    for i, t in enumerate(r.tensors):
        if t.name in names:
            x = gguf.quants.dequantize(t.data, QT(t.tensor_type))
            q = gguf.quants.quantize(x, QT.Q8_0)
            y = gguf.quants.dequantize(q, QT.Q8_0)
            err = float(np.sqrt(np.mean((y - x) ** 2) / max(1e-30, np.mean(x ** 2))))
            worst = max(worst, err)
            w.write_tensor_data(q)
        else:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        if i % 100 == 0:
            print(f"  {i}/{len(r.tensors)}  {time.time() - t0:.0f}s", flush=True)
    w.close()
    os.rename(a.dst + ".part", a.dst)
    print(f"done in {time.time() - t0:.0f}s; worst relative rms error of the re-encode {worst:.2e}")


if __name__ == "__main__":
    main()
