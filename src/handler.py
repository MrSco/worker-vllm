import logging
import traceback
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("handler")

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    job_id = job.get("id", "unknown")
    logger.info("[Job %s] Received new job request", job_id)

    try:
        job_input = JobInput(job["input"])
        input_data = job["input"]

        # Fallback: RunPod OpenAI proxy may send raw request body without openai_route.
        # When input has messages (OpenAI chat format), route to OpenAI engine for correct handling.
        if not job_input.openai_route and isinstance(input_data.get("messages"), list):
            logger.info("[Job %s] No openai_route provided but messages detected — routing as /v1/chat/completions", job_id)
            job_input.openai_route = "/v1/chat/completions"
            job_input.openai_input = input_data

        if job_input.openai_route:
            logger.info("[Job %s] Using OpenAI engine (route: %s, stream: %s)", job_id, job_input.openai_route, input_data.get("stream", False))
        else:
            logger.info("[Job %s] Using vLLM engine (stream: %s, request_id: %s)", job_id, job_input.stream, job_input.request_id)

        engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)

        batch_count = 0
        async for batch in results_generator:
            batch_count += 1
            yield batch

        logger.info("[Job %s] Completed — yielded %d batch(es)", job_id, batch_count)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[Job %s] Unhandled error — finishing job with error to prevent retry.\n%s", job_id, tb)
        yield {"error": {"message": f"{type(e).__name__}: {e}", "traceback": tb}}

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)