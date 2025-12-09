#!/bin/bash
alias gg='git fetch && git checkout origin/zejun/plugin_for_atom_1223'
alias tt='bash ./test.tp8.model.impl.sh'
alias cc='bash ./curl.sh'

export VLLM_ATTENTION_BACKEND=CUSTOM

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export VLLM_ROCM_USE_AITER_TRITON_ROPE=0

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/data/pretrained-models/Qwen3-0.6B

vllm serve $model_path \
    --host localhost \
    --port 9090 \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.1 \
    --async-scheduling \
    --load-format fastsafetensors \
    --no-enable-chunked-prefill \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --model-impl atom \
    2>&1 | tee log.serve.log &
