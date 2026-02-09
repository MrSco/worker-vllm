import os
import sys
import subprocess
import multiprocessing

# === CUDA Diagnostics (remove after debugging) ===
print("=" * 60, flush=True)
print("CUDA DIAGNOSTIC - Test 1: Basic CUDA init", flush=True)
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}", flush=True)
    print(f"  CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"  Device count: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"  Device name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  CUDA version: {torch.version.cuda}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print("-" * 60, flush=True)
print("CUDA DIAGNOSTIC - Test 2: CUDA in child process (fork)", flush=True)

def _test_cuda_child():
    try:
        import torch
        torch.cuda.init()
        print(f"  Child CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"  Child device: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as e:
        print(f"  Child FAILED: {e}", flush=True)
        sys.exit(1)

p = multiprocessing.Process(target=_test_cuda_child)
p.start()
p.join()
print(f"  Child exit code: {p.exitcode}", flush=True)

print("-" * 60, flush=True)
print("CUDA DIAGNOSTIC - Test 3: CUDA in child process (spawn)", flush=True)
ctx = multiprocessing.get_context("spawn")
p2 = ctx.Process(target=_test_cuda_child)
p2.start()
p2.join()
print(f"  Spawn child exit code: {p2.exitcode}", flush=True)

print("-" * 60, flush=True)
print("CUDA DIAGNOSTIC - Test 4: LD_LIBRARY_PATH and compat check", flush=True)
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(not set)')}", flush=True)
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}", flush=True)
print(f"  CUDA_MODULE_LOADING: {os.environ.get('CUDA_MODULE_LOADING', '(not set)')}", flush=True)
subprocess.run(["ls", "-la", "/etc/ld.so.conf.d/"], check=False)
subprocess.run(["ldconfig", "-p"], check=False, stdout=subprocess.PIPE)  # just check it runs
print("=" * 60, flush=True)
print("CUDA DIAGNOSTICS COMPLETE - proceeding with handler", flush=True)
print("=" * 60, flush=True)

# === End diagnostics ===

import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    job_input = JobInput(job["input"])
    input_data = job["input"]
    # Fallback: RunPod OpenAI proxy may send raw request body without openai_route.
    # When input has messages (OpenAI chat format), route to OpenAI engine for correct handling.
    if not job_input.openai_route and isinstance(input_data.get("messages"), list):
        job_input.openai_route = "/v1/chat/completions"
        job_input.openai_input = input_data
    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)