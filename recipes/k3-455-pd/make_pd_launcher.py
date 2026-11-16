#!/usr/bin/env python3
"""make_pd_launcher.py <in launcher> <out launcher> <ATOM commit>: a copy of the image kit's launcher
(/root/.cache/k3_server_dspark.sh, make_launcher.sh) that serves the PD ATOM tree /root/.cache/atom-pd at <commit> and,
with K3_KV_TRANSFER_CONFIG set, passes it as --kv-transfer-config. Unset, it is the original command apart from the
ATOM tree."""
import pathlib
import sys

src, dst, commit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
s = src.read_text()
kv_block = '''# --- k3pd: P/D. K3_KV_TRANSFER_CONFIG is the JSON of --kv-transfer-config (moriio over the UALink fabric)
KV_ARGS=()
[ -z "${K3_KV_TRANSFER_CONFIG:-}" ] || KV_ARGS=(--kv-transfer-config "$K3_KV_TRANSFER_CONFIG")

# Online FP8 for the checkpoint's BF16 layers'''
edits = [
    ("ATOM_COMMIT=746624d4c3c2b2593f8ff832ff01f1118f1bd177\n", f"ATOM_COMMIT={commit}\n"),
    ("ATOM_DIR=${K3_ATOM_DIR:-/root/.cache/atom-gyu-k3-455-tp8}\n",
     "ATOM_DIR=/root/.cache/atom-pd   # k3pd: always the PD tree (make_pd_launcher.sh)\n"),
    ("# Online FP8 for the checkpoint's BF16 layers", kv_block),
    ('"${MODE_ARGS[@]}" "${SPEC_ARGS[@]}" "${BLOCK_ARGS[@]}"',
     '"${MODE_ARGS[@]}" "${SPEC_ARGS[@]}" "${BLOCK_ARGS[@]}" "${KV_ARGS[@]}"'),
    ('max-num-seqs $MAX_SEQS block-size ${K3_BLOCK_SIZE:-16}"',
     'max-num-seqs $MAX_SEQS block-size ${K3_BLOCK_SIZE:-16} kv ${K3_KV_TRANSFER_CONFIG:-none}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"{src}: expected one {old[:70]!r}, found {s.count(old)}")
    s = s.replace(old, new)
tmp = dst.with_name(dst.name + ".new")
tmp.write_text(s)
tmp.chmod(0o755)
tmp.replace(dst)
print(f"{dst}: ATOM {commit[:9]} at /root/.cache/atom-pd, K3_KV_TRANSFER_CONFIG knob")
