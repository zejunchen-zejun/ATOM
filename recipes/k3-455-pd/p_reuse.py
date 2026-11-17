#!/usr/bin/env python3
"""p_reuse.py <run dir> [--hb 1024] [--window 48] [--example 1]: what P's prefill tokens are spent on, turn by turn.

P logs every arrival ("Request N arrived, input tokens: L") and every prefill batch ("done: [...]", "new: [...]",
"req_ids: (...)": per request, the prefix-cache hit and the tokens computed). The client's events.jsonl gives each
sample's conversation and turn (1, 3, 5, ...). Samples map to P request ids in issue order, the last runs' samples
being the last arrivals; the router reorders requests sent at nearly the same moment, so a failed pair is repaired
by swapping with a sample at most --window places away whose request does fit.

Only full hash blocks are indexed (block_size x dcp = 128 x 8 = 1024 tokens on K3 DCP8), and P never sees D's
output, so a turn hits at most the previous turn's prompt rounded down to the hash block. An exact pair is one that
hits exactly that; its computed tokens split into
  tail    the previous prompt's last partial hash block, recomputed (L_prev - floor(L_prev / hb) * hb)
  fresh   the previous turn's output plus the new message (L - L_prev)
A pair that hits less is "below cap": eviction, a declined checkpoint, or a pair the mapping still gets wrong.
"""
import argparse
import collections
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--hb", type=int, default=1024)
ap.add_argument("--window", type=int, default=48)
ap.add_argument("--example", type=int, default=1, help="conversations to print turn by turn")
a = ap.parse_args()
run, hb = Path(a.run), a.hb

ARR = re.compile(rb"Request (\d+) arrived, input tokens: (\d+)")
SCH = re.compile(rb"Scheduled prefill batch: \d+ reqs, \d+ new tokens \(done: \[([^\]]*)\], new: \[([^\]]*)\]\), "
                 rb"req_ids: \(([^)]*)\)")
L, hit, new = {}, {}, collections.Counter()
for line in open(run / "logs" / "P0.log", "rb"):
    line = line.replace(b"\0", b"")
    m = ARR.search(line)
    if m:
        L[int(m[1])] = int(m[2])
        continue
    m = SCH.search(line)
    if m:
        ds, ns, ids = ([int(x) for x in g.split(b",") if x.strip()] for g in m.groups())
        for d, n, r in zip(ds, ns, ids):
            hit.setdefault(r, d)  # a chunked prefill's later chunks report their own progress as done
            new[r] += n

ISS = re.compile(r'"sample.issued","timestamp_ns":(\d+),"sample_uuid":"\w*","conversation_id":"([^"]*)","turn":(\d+)')
runs = []
for ev in sorted(run.glob("*/events.jsonl"), key=lambda p: p.stat().st_mtime):
    s = []
    with open(ev, "rb") as f:
        for raw in f:  # an issued event carries the whole prompt; the fields needed are at its start
            m = ISS.search(raw[:400].decode("utf-8", "replace"))
            if m:
                s.append((int(m[1]), m[2], int(m[3])))
    s.sort()
    runs.append((ev.parent.name, s))
rids = sorted(L)
nsamp = sum(len(s) for _, s in runs)
print(f"P arrivals {len(rids)}, client samples {nsamp} in {len(runs)} runs, {len(rids) - nsamp} earlier (gate)")
base = len(rids) - nsamp

for name, s in runs:
    n = len(s)
    rid = [rids[base + i] for i in range(n)]
    base += n
    pred, last = [-1] * n, {}
    for i, (_, conv, _) in enumerate(s):
        pred[i], last[conv] = last.get(conv, -1), i

    def exact(i):
        r, q = rid[i], rid[pred[i]]
        return r in hit and hit[r] == L[q] // hb * hb and L[r] > L[q]

    pairs = [i for i in range(n) if pred[i] >= 0]
    before = sum(exact(i) for i in pairs)
    for _ in range(4):
        moved = 0
        for i in pairs:
            if exact(i):
                continue
            q = rid[pred[i]]
            for k in sorted(range(max(0, i - a.window), min(n, i + a.window + 1)), key=lambda k: abs(k - i)):
                r = rid[k]
                if (k != i and r in hit and hit[r] == L[q] // hb * hb and L[r] > L[q]
                        and not (pred[k] >= 0 and exact(k))):
                    rid[i], rid[k] = r, rid[i]
                    moved += 1
                    break
        if not moved:
            break
    tot = collections.Counter()
    for i in pairs:
        r, q = rid[i], rid[pred[i]]
        cap = L[q] // hb * hb
        if exact(i):
            tot["pairs exact"] += 1
            tot["tail"] += L[q] - cap
            tot["fresh"] += L[r] - L[q]
        elif L[r] > L[q] and r in hit and hit[r] < cap:
            tot["pairs below cap"] += 1
            tot["below cap"] += cap - hit[r]
        else:
            tot["pairs unmatched"] += 1
    computed = sum(new[r] for r in rid)
    first = sum(new[rid[i]] for i in range(n) if pred[i] < 0)
    print(f"\n{name}: {n} requests in {sum(1 for p in pred if p < 0)} conversations, P computed {computed:,} tokens "
          f"(first turns {100 * first / computed:.1f}%)")
    print(f"  turn pairs {len(pairs)}: exact {tot['pairs exact']} ({100 * tot['pairs exact'] / len(pairs):.1f}%; "
          f"{100 * before / len(pairs):.1f}% before repair), below cap {tot['pairs below cap']}, "
          f"unmatched {tot['pairs unmatched']}")
    print(f"  exact pairs computed tail {tot['tail']:,} + fresh {tot['fresh']:,}: "
          f"tail {100 * tot['tail'] / max(1, tot['tail'] + tot['fresh']):.1f}%")
    print(f"  below cap (upper bound on prefix lost to eviction): {tot['below cap']:,} tokens, "
          f"{100 * tot['below cap'] / computed:.1f}% of P computed")
    byconv = collections.defaultdict(list)
    for i, (_, conv, turn) in enumerate(s):
        byconv[conv].append(i)
    shown = 0
    for conv, idx in byconv.items():
        if shown >= a.example:
            break
        run_ok = [i for i in idx if pred[i] >= 0 and exact(i)]
        if len(run_ok) < 6 or len(run_ok) < len(idx) - 1:
            continue
        shown += 1
        print(f"  e.g. {conv}:")
        for i in idx[:8]:
            r = rid[i]
            line = f"    turn {s[i][2]:3d} req {r:5d}: input {L[r]:6d}  hit {hit.get(r, 0):6d}  computed {new[r]:5d}"
            if pred[i] >= 0 and exact(i):
                q = L[rid[pred[i]]]
                line += f" = tail {q - q // hb * hb:4d} (previous input {q} cached as {q // hb * hb}) + fresh {L[r] - q}"
            print(line)
