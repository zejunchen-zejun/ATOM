#!/usr/bin/env python3
"""trace_top.py <trace file or directory> [--top N]: where the GPU time of a torch-profiler trace goes.

Sums the durations of GPU events (kernels, memcpy, memset) of every *.json / *.json.gz trace under the path, per rank
file and over all of them, and prints the share of each category (MoE, MLA attention, KDA / linear attention, dense
GEMM, communication, other) and the top kernels by total time. Categories come from kernel-name patterns; check the
top list for names that land in "other". Kernels replayed inside CUDA graphs are invisible to the profiler (see
profile_pd.sh).
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = [  # first match wins
    ("comm", r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|nccl|rccl|lamport|cross_device|symm|mori|"
             r"one_shot|two_shot|quick_reduce|ar_2s|ar_1s"),
    ("moe", r"moe|grouped_gemm|expert|a4w4|mxfp4|a8w4|topk|sort_tok|fused_route|router"),
    ("kda", r"kda|gdn|chunk_delta|delta_attn|flash_kda|gla|ssm|replay|conv1d|causal_conv|chunk_scaled|wy_fast"),
    ("mla_attn", r"mla|fmha|flash_attn|flashattn|attn|attention|paged|ps1|softmax_lse|merge_attn|reduce_attn"),
    ("gemm", r"gemm|matmul|cutlass|hipblaslt|cijk|wvsplitk|ck_tile|bf16_gemm|fp8_gemm|blockscale"),
    ("memcpy", r"memcpy|memset|copy_|cat_|index_|gather|scatter"),
]
CAT_RE = [(c, re.compile(p, re.I)) for c, p in CATEGORIES]


def category(name: str) -> str:
    for c, rx in CAT_RE:
        if rx.search(name):
            return c
    return "other"


def events(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    for e in evs:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset"):
            yield e.get("name", "?"), float(e.get("dur", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    root = Path(a.path)
    files = sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.name.endswith((".json", ".json.gz")))
    if not files:
        sys.exit(f"no traces under {root}")
    total_k, total_c, per_file = Counter(), Counter(), []
    for f in files:
        k = Counter()
        for name, dur in events(f):
            short = re.sub(r"<.*", "", name)[:110]
            k[short] += dur
        if not k:
            continue
        c = Counter()
        for n, d in k.items():
            c[category(n)] += d
        per_file.append((f, sum(k.values()), c))
        total_k.update(k)
        total_c.update(c)
    tot = sum(total_c.values())
    print(f"{len(per_file)} trace files with GPU events, {tot / 1e6:.2f} s of GPU time in total")
    for f, t, c in per_file:
        shares = "  ".join(f"{cat} {c[cat] / t * 100:4.1f}%" for cat, _ in CATEGORIES + [("other", "")] if c[cat])
        print(f"  {f.relative_to(root) if root.is_dir() else f.name}: {t / 1e6:.2f} s | {shares}")
    print("all ranks: " + "  ".join(f"{cat} {total_c[cat] / tot * 100:.1f}%" for cat, _ in CATEGORIES + [("other", "")]))
    print(f"top {a.top} kernels (all ranks):")
    for n, d in total_k.most_common(a.top):
        print(f"  {d / tot * 100:5.1f}%  {d / 1e6:8.3f} s  [{category(n)}] {n}")


if __name__ == "__main__":
    main()
