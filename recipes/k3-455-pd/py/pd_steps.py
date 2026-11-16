#!/usr/bin/env python3
"""pd_steps.py <run dir> [--from S] [--to S]: what P and D did in a real 1P1D run, from the engine cores' step logs
(py/sitecustomize.py; run_pd.sh copies them to <run dir>/logs/<tag>_P and <tag>_D).

The span is the measured window (<run dir>/window, epoch start and end) from start + S1 (default 60: past the opening
burst, every user sending its first turn) to start + S2 (default 600: the 10-minute issue cap). Per instance: share of
the span with a forward running (busy); P: prefill steps, new and cached tokens per second, prefix hit rate, new tokens
per step; D: decode steps, output tokens per second (one per running request per step: no DSpark), its batch and the
step cycle (start to next start) by batch size.
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--from", dest="s1", type=float, default=60.0)
ap.add_argument("--to", dest="s2", type=float, default=600.0)
a = ap.parse_args()
t0 = float(open(os.path.join(a.run, "window")).read().split()[0])
lo, hi = t0 + a.s1, t0 + a.s2
span = hi - lo


def steps(role):
    out = []
    for f in glob.glob(os.path.join(a.run, "logs", f"*_{role}", "steps_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if lo <= r["t0"] < hi:
                out.append(r)
    out.sort(key=lambda r: r["t0"])
    return out


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


print(f"span {a.s1:.0f}-{a.s2:.0f} s of the measured window ({span:.0f} s)")
P = steps("P")
if P:
    pre = [r for r in P if r["k"] == "p"]
    new = np.array([r.get("new", 0) for r in pre])
    cached = np.array([r.get("cached", 0) for r in pre])
    busy = sum(r["dt"] for r in P) / span
    print(f"P: {len(P)} steps ({len(pre)} prefill), busy {100 * busy:.0f}% | new {new.sum() / span:,.0f} tok/s, "
          f"cached {cached.sum() / span:,.0f} tok/s, hit {100 * cached.sum() / max(1, new.sum() + cached.sum()):.1f}% | "
          f"new per step p50/p90 {pct(new, 50):,.0f}/{pct(new, 90):,.0f} | requests per step p50 "
          f"{pct([r['n'] for r in pre], 50):.0f} | step p50 {1e3 * pct([r['dt'] for r in pre], 50):.0f} ms")
D = steps("D")
if D:
    dec = [r for r in D if r["k"] == "d"]
    b = np.array([r["n"] for r in dec])
    busy = sum(r["dt"] for r in D) / span
    print(f"D: {len(D)} steps ({len(dec)} decode), busy {100 * busy:.0f}% | output {b.sum() / span:,.0f} tok/s "
          f"({b.sum() / span / 16:,.0f} per GPU of 16) | batch p10/p50/p90 {pct(b, 10):.0f}/{pct(b, 50):.0f}/"
          f"{pct(b, 90):.0f} | context per request p50 {pct([r['ctx'] / max(1, r['n']) for r in dec], 50):,.0f}")
    cyc = [(dec[i]["n"], dec[i + 1]["t0"] - dec[i]["t0"]) for i in range(len(dec) - 1)]
    for blo, bhi in ((1, 127), (128, 255), (256, 383), (384, 511), (512, 512)):
        c = [t for n, t in cyc if blo <= n <= bhi]
        if c:
            print(f"   B {blo}-{bhi}: {len(c)} steps, cycle p50 {1e3 * pct(c, 50):.1f} ms")
