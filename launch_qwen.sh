set -x
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export AITER_ROPE_FUSED_QKNORM=1

# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
# model_path=/mnt/raid0/pretrained_model/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
# model_path=/mnt/raid0/pretrained_model/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
model_path=/mnt/raid0/pretrained_model/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8


TORCHINDUCTOR_COMPILE_THREADS=128 CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m sglang.launch_server \
    --model-path $model_path \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --expert-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.7 \
    --disable-cuda-graph \
    --model-impl atom \
    --page-size 1024 \
    2>&1 | tee log.serve.log

#  curl -X POST "http://localhost:8000/v1/completions" \
#      -H "Content-Type: application/json" \
#      -d '{
#          "prompt": "The capital of China", "temperature": 0, "top_p": 1, 
#          "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, 
#          "stream": false, "ignore_eos": false, "n": 1, "seed": 123
#  }'