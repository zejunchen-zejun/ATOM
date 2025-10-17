import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from atom import LLMEngine, SamplingParams
from atom.config import CompilationConfig


# Request models
class CompletionRequest(BaseModel):
    model: str = "atom"
    prompt: str | List[str]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[str | List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "atom"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[str | List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


# Global engine and tokenizer
llm_engine: Optional[LLMEngine] = None
tokenizer: Optional[Any] = None
# Lock to serialize generation calls
generation_lock: Optional[asyncio.Lock] = None
# Log file path
log_file_path: Optional[str] = None


def parse_size_list(size_str: str) -> List[int]:
    import ast
    try:
        return ast.literal_eval(size_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error list size: {size_str}") from e


def log_request_response(request_id: str, prompts: List[str], outputs: List[Dict], endpoint: str):
    """Log request inputs and outputs to JSONL file"""
    if log_file_path is None:
        return
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "endpoint": endpoint,
        "prompts": prompts,
        "outputs": [{"text": out["text"]} for out in outputs]
    }
    
    try:
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")


def init_engine(args):
    global llm_engine, tokenizer, generation_lock, log_file_path
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # LLMEngine needs a separate port for distributed communication
    engine_port = args.port + 1000  # e.g., if API port is 8000, engine uses 9000
    llm_engine = LLMEngine(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        port=engine_port,
        torch_profiler_dir=args.torch_profiler_dir,
        compilation_config=CompilationConfig(
            level=args.level,
            cudagraph_capture_sizes=parse_size_list(args.cudagraph_capture_sizes)
        )
    )
    generation_lock = asyncio.Lock()
    
    # Set log file path if enabled
    if args.log_requests:
        log_file_path = args.log_file
        print(f"Request/Response logging enabled: {log_file_path}")


app = FastAPI(title="Atom OpenAI-Compatible API")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if llm_engine is None or generation_lock is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop_strings=request.stop,
    )
    
    # Use lock to serialize generation calls and run in thread pool to avoid blocking
    async with generation_lock:
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, 
            llm_engine.generate, 
            prompts, 
            sampling_params
        )
    
    # Log request and response
    log_request_response(request_id, prompts, outputs, "/v1/completions")
    
    choices = []
    for i, output in enumerate(outputs):
        choices.append({
            "text": output["text"],
            "index": i,
            "logprobs": None,
            "finish_reason": "stop"
        })
    
    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": sum(len(tokenizer.encode(p)) for p in prompts),
            "completion_tokens": sum(len(tokenizer.encode(c["text"])) for c in choices),
            "total_tokens": sum(len(tokenizer.encode(p)) for p in prompts) + sum(len(tokenizer.encode(c["text"])) for c in choices)
        }
    })


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if llm_engine is None or tokenizer is None or generation_lock is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # Convert chat messages to prompt
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        # Fallback: simple concatenation
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop_strings=request.stop,
    )
    
    # Use lock to serialize generation calls and run in thread pool to avoid blocking
    async with generation_lock:
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            llm_engine.generate,
            [prompt],
            sampling_params
        )
    
    # Log request and response
    log_request_response(request_id, [prompt], outputs, "/v1/chat/completions")
    
    completion_text = outputs[0]["text"] if outputs else ""
    
    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion_text,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(completion_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(completion_text))
        }
    })


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "atom",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "atom"
        }]
    })


@app.get("/health")
async def health():
    return {"status": "healthy", "engine_ready": llm_engine is not None}


def main():
    parser = argparse.ArgumentParser(description="Atom OpenAI-Compatible Serving")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port (engine will use port+1000)")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--kv_cache_dtype", choices=["bf16", "fp8"], type=str, default="bf16")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager mode")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching")
    parser.add_argument("--cudagraph-capture-sizes", type=str, default="[1,2,4,8,16]")
    parser.add_argument("--level", type=int, default=3, help="Compilation level")
    parser.add_argument("--torch-profiler-dir", type=str, default=None)
    parser.add_argument("--log-requests", action="store_true", help="Enable request/response logging")
    parser.add_argument("--log-file", type=str, default="serving_requests.jsonl", help="Path to log file for requests/responses")
    
    args = parser.parse_args()
    
    print(f"Initializing engine with model: {args.model}")
    init_engine(args)
    print("Engine initialized successfully")
    
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
