#!/bin/bash
echo "run vllm"
export VLLM_ATTENTION_BACKEND=CUSTOM

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export VLLM_ROCM_USE_AITER_TRITON_ROPE=0

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507-FP8

# TODO: why need max model len
vllm serve $model_path \
    --host localhost \
    --port 9090 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.8 \
    --async-scheduling \
    --load-format fastsafetensors \
    --no-enable-chunked-prefill \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --model-impl atom \
    --max-model-len 16384 \
    2>&1 | tee log.serve.log &
