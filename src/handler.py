import os
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