import cProfile
import os
import pstats
import time
from random import randint, seed

from transformers import AutoTokenizer

from atom import LLMEngine, SamplingParams
from atom.config import CompilationConfig

# A very long prompt, total number of tokens is about 15k.
# LONG_PROMPT = ["You are an expert in large language models, aren't you?"] * 256
# LONG_PROMPT = " ".join(LONG_PROMPT)
LONG_PROMPT = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 64


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/mnt/raid0/lirong/Qwen3-0.6B/")
    llm = LLMEngine(
        path,
        enforce_eager=False,
        max_model_len=4096,
        port=12345,
        torch_profiler_dir="./log",
        compilation_config=CompilationConfig(
            level=3,
            # cudagraph_capture_sizes=[1, 2, 4, 8] + list(range(16, 512 + 1, 16))
        ),
    )

    # tokenizer = AutoTokenizer.from_pretrained(path)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]

    # sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16)
        for _ in range(num_seqs)
    ]

    print("------start generating------")
    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         enable_thinking=True,
    #     )
    #     for prompt in prompts
    # ]

    outputs = llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    outputs = llm.generate(LONG_PROMPT, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
