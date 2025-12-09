#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_ROCM_USE_ATOM_PLUGIN=1

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/raid/data/model/Qwen3-235B-A22B-Instruct-2507-FP8

vllm serve $model_path \
    --host localhost \
    --port 9090 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-num-batched-tokens 20000 \
    --kv-cache-dtype "fp8" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.6 \
    --enforce-eager \
    --async-scheduling \
    --load-format fastsafetensors \
    2>&1 | tee log.serve.log &

