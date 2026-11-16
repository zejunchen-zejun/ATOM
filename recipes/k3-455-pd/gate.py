"""Accuracy gate for a running Kimi-K3 server. Run it before any benchmark: the worst failures are silent.

Fails on: all '!' (A0 silicon without translation, or NaN decode), one repeated token, empty answers, and a chat
'17 * 23' whose content is not 391 (K3 reasons first; the answer must land in `content`). Warns when a decoded
continuation and a prefilled one differ at temperature 0. Stdlib only.
Usage: python3 /opt/k3/gate.py [base url, default http://127.0.0.1:8000]
"""

import json
import sys
import urllib.request

EP = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = "moonshotai/Kimi-K3"
fails, warns = [], []


def call(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(EP + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def complete(prompt, max_tokens):
    c = call("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
    return c["choices"][0]["text"], c["choices"][0].get("finish_reason")


# 1. the server answers, under the name every client must use
print("models:", [m["id"] for m in call("/v1/models")["data"]])

# 2. not empty, not all '!', not one token repeated
for p in ["The capital of France is", "1 + 1 =", "The chemical symbol for gold is", "Water boils at"]:
    text, finish = complete(p, 24)
    print(f"{p!r} -> {text!r} ({finish})")
    s = text.strip()
    if not s or set(s) <= {"!"}:
        fails.append(f"{p!r}: empty or all '!'")
    elif len(s.split()) >= 6 and len(set(s.split())) == 1:
        fails.append(f"{p!r}: one token repeated")

# 3. decode vs prefill at temperature 0: continue from the midpoint of a decoded answer by prefilling it
p = "Write one sentence about the history of the Eiffel Tower."
full, _ = complete(p, 48)
cut = full.rfind(" ", 0, len(full) // 2)
if cut > 0:
    tail = full[cut:]
    cont, _ = complete(p + full[:cut], 24)
    same = tail.split()[:5] == cont.split()[:5]
    print(f"decode vs prefill: {'match' if same else 'MISMATCH'}\n  decoded:   {tail[:90]!r}\n  prefilled: {cont[:90]!r}")
    if not same:
        warns.append("decode/prefill continuations differ in the first 5 words: inspect by hand "
                     "(a different token split at the cut can also cause this)")

# 4. reasoning model: the chain of thought goes to reasoning_content, the answer to content
c = call("/v1/chat/completions", {"model": MODEL, "max_tokens": 4096, "temperature": 0,
                                  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]})
msg, finish = c["choices"][0]["message"], c["choices"][0].get("finish_reason")
print(f"chat: content={msg.get('content')!r} | reasoning {len(msg.get('reasoning_content') or '')} chars | finish={finish}")
if "391" not in (msg.get("content") or ""):
    (warns if finish == "length" else fails).append(f"chat: '391' not in content (finish={finish})")

for w in warns:
    print("WARN", w)
print("GATE " + ("PASS" if not fails else "FAIL\n  " + "\n  ".join(fails)))
sys.exit(1 if fails else 0)
