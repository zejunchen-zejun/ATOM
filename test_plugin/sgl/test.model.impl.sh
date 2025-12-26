#!/bin/bash
echo "run sgl"
export CUDA_VISIBLE_DEVICES=7
# export AMD_SERIALIZE_KERNEL=3

export SAFETENSORS_FAST_GPU=1

rm -rf /root/.cache/

model_path=/data/pretrained-models/Qwen3-0.6B

python3 -m sglang.launch_server \
    --model-path $model_path \
    --host localhost \
    --port 9090 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.1 \
    --model-impl atom \
    --disable-cuda-graph \
    --no-enable-torch-compile \
    2>&1 | tee log.serve.log &
