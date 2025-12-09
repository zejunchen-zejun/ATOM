#!/bin/bash

export VLLM_ROCM_USE_ATOM_PLUGIN=1

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/raid/data/model/Qwen3-0.6B

vllm serve $model_path \
    --host localhost \
    --port 9090 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype "fp8" \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.6 \
    --enforce-eager \
    --async-scheduling \
    --load-format fastsafetensors \
    2>&1 | tee log.serve.log &

