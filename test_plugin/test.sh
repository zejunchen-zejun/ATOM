#!/bin/bash
alias gg='git fetch && git checkout origin/zejun/plugin_for_atom'
export CUDA_VISIBLE_DEVICES=7
export VLLM_ROCM_USE_ATOM_PLUGIN=1
export VLLM_ATTENTION_BACKEND=CUSTOM

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export AMD_SERIALIZE_KERNEL=3
export ROCM_DEBUG_AGENT_OPTIONS="--all"
export HSA_TOOLS_LIB=/opt/rocm-7.1.0/lib/librocm-debug-agent.so.2
export TORCH_LOGS="+dynamo"

export VLLM_ROCM_USE_AITER_TRITON_ROPE=0

# cache dir
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=/raid/data/model/Qwen3-0.6B

vllm serve $model_path \
    --host localhost \
    --port 9090 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.1 \
    --compilation-config '{"cudagraph_mode": "None"}' \
    --async-scheduling \
    --load-format fastsafetensors \
    --no-enable-chunked-prefill \
    2>&1 | tee log.serve.log &
