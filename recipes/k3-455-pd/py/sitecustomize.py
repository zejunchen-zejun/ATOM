# K3 PD emulation step log (kit-only, not in the image or any PR). Active only when K3_STEPLOG names a directory.
# When atom.model_engine.engine_core is imported, EngineCore._process_engine_step_inner is wrapped: every step that ran
# a forward writes one JSON line (start time, duration, the gap since the previous step's end, and the batch the
# scheduler built: prefill or decode, requests, tokens, context, cached tokens, prefill attention work).
# One file per process: <K3_STEPLOG>/steps_<host>_<pid>.jsonl; only the engine core process steps, so only it writes.
import os
import sys

_DIR = os.environ.get("K3_STEPLOG")
_TARGET = "atom.model_engine.engine_core"


def _summary(b):
    rec = {"k": "p" if getattr(b, "total_seqs_num_prefill", 0) else "d", "n": len(b.req_ids),
           "tok": int(b.total_tokens_num), "ctx": int(b.context_lens.sum()), "spec": int(getattr(b, "num_spec_step", 0))}
    if rec["k"] == "p":
        new = [int(x) for x in b.num_scheduled_tokens]
        cached = [int(x) for x in b.num_cached_tokens]
        rec["new"] = sum(new)
        rec["cached"] = sum(cached)
        # attention work of the chunk: sum over sequences of new x (cached + new / 2)
        rec["att"] = int(sum(n * (c + n / 2) for n, c in zip(new, cached)))
        rec["nd"] = int(getattr(b, "total_seqs_num_decode", 0))
    return rec


def _patch(mod):
    import atexit
    import json
    import socket
    import time

    ec = getattr(mod, "EngineCore", None)
    if ec is None or getattr(ec, "_k3pd_patched", False):
        return
    orig = ec._process_engine_step_inner
    st = {"f": None, "end": None, "n": 0, "err": 0, "flushed": 0.0}

    def step(self):
        sch = self.scheduler
        if not getattr(sch, "_k3pd_wrapped", False):
            osched = sch.schedule

            def sched(*a, **k):
                r = osched(*a, **k)
                sch._k3pd_last = r
                return r

            sch.schedule = sched
            sch._k3pd_wrapped = True
        sch._k3pd_last = None
        t0 = time.time()
        ret = orig(self)
        t1 = time.time()
        r = sch._k3pd_last
        try:
            if r is not None and r[0] is not None and len(r[0].req_ids) > 0:
                rec = _summary(r[0])
                rec["t0"] = round(t0, 6)
                rec["dt"] = round(t1 - t0, 6)
                rec["gap"] = None if st["end"] is None else round(t0 - st["end"], 6)
                if st["f"] is None:
                    os.makedirs(_DIR, exist_ok=True)
                    st["f"] = open(os.path.join(_DIR, f"steps_{socket.gethostname()}_{os.getpid()}.jsonl"), "a",
                                   buffering=1 << 20)
                    atexit.register(st["f"].flush)
                st["f"].write(json.dumps(rec, separators=(",", ":")) + "\n")
                st["n"] += 1
                if t1 - st["flushed"] > 2.0:  # the server is stopped with kill -9: keep at most ~2 s unwritten
                    st["f"].flush()
                    st["flushed"] = t1
        except Exception as e:  # never let the log break serving
            st["err"] += 1
            if st["err"] <= 3:
                print(f"[k3pd steplog] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        st["end"] = t1
        return ret

    ec._process_engine_step_inner = step
    ec._k3pd_patched = True
    print(f"[k3pd steplog] EngineCore patched in pid {os.getpid()}, logging to {_DIR}", file=sys.stderr, flush=True)


if _DIR:
    import importlib.abc

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader_exec = spec.loader.exec_module

            def exec_module(module):
                loader_exec(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

# This file shadows the image's own sitecustomize (Ubuntu's: install the apport exception hook if apport is there);
# run that one too.
_ORIG = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_ORIG):
    try:
        exec(compile(open(_ORIG).read(), _ORIG, "exec"), {"__name__": "sitecustomize_orig"})
    except Exception:
        pass
