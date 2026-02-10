import os
import logging
import json
import asyncio
import traceback

from dotenv import load_dotenv
from typing import AsyncGenerator, Optional
import time

from vllm import AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels


from utils import DummyRequest, JobInput, BatchSize, create_error_response
from constants import DEFAULT_MAX_CONCURRENCY, DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE_GROWTH_FACTOR, DEFAULT_MIN_BATCH_SIZE
from tokenizer import TokenizerWrapper
from engine_args import get_engine_args

logger = logging.getLogger("engine")

class vLLMEngine:
    def __init__(self, engine = None):
        load_dotenv() # For local development
        logger.info("Initializing vLLMEngine...")
        self.engine_args = get_engine_args()
        logger.info("Engine args loaded: model=%s, tokenizer_mode=%s", self.engine_args.model, self.engine_args.tokenizer_mode)
        
        # Initialize vLLM engine first
        self.llm = self._initialize_llm() if engine is None else engine.llm
        
        # Only create custom tokenizer wrapper if not using mistral tokenizer mode
        # For mistral models, let vLLM handle tokenizer initialization
        if self.engine_args.tokenizer_mode != 'mistral':
            logger.info("Loading custom tokenizer wrapper for model: %s", self.engine_args.tokenizer or self.engine_args.model)
            self.tokenizer = TokenizerWrapper(self.engine_args.tokenizer or self.engine_args.model, 
                                              self.engine_args.tokenizer_revision, 
                                              self.engine_args.trust_remote_code)
        else:
            # For mistral models, we'll get the tokenizer from vLLM later
            logger.info("Using mistral tokenizer mode — deferring tokenizer init to vLLM")
            self.tokenizer = None
            
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
        self.default_batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        self.batch_size_growth_factor = int(os.getenv("BATCH_SIZE_GROWTH_FACTOR", DEFAULT_BATCH_SIZE_GROWTH_FACTOR))
        self.min_batch_size = int(os.getenv("MIN_BATCH_SIZE", DEFAULT_MIN_BATCH_SIZE))
        logger.info("vLLMEngine ready (max_concurrency=%d, default_batch_size=%d)", self.max_concurrency, self.default_batch_size)

    def _get_tokenizer_for_chat_template(self):
        """Get tokenizer for chat template application"""
        if self.tokenizer is not None:
            return self.tokenizer
        else:
            # For mistral models, get tokenizer from vLLM engine
            # This is a fallback - ideally chat templates should be handled by vLLM directly
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.engine_args.tokenizer or self.engine_args.model,
                    revision=self.engine_args.tokenizer_revision or "main",
                    trust_remote_code=self.engine_args.trust_remote_code
                )
                # Create a minimal wrapper
                class MinimalTokenizerWrapper:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
                        self.has_chat_template = bool(self.tokenizer.chat_template) or bool(self.custom_chat_template)
                        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
                            self.tokenizer.chat_template = self.custom_chat_template
                    
                    def apply_chat_template(self, input):
                        if isinstance(input, list):
                            if not self.has_chat_template:
                                raise ValueError(
                                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                                )
                        elif isinstance(input, str):
                            input = [{"role": "user", "content": input}]
                        else:
                            raise ValueError("Input must be a string or a list of messages")
                        
                        return self.tokenizer.apply_chat_template(
                            input, tokenize=False, add_generation_prompt=True
                        )
                
                return MinimalTokenizerWrapper(tokenizer)
            except Exception as e:
                logging.error(f"Failed to create fallback tokenizer: {e}")
                raise e

    def dynamic_batch_size(self, current_batch_size, batch_size_growth_factor):
        return min(current_batch_size*batch_size_growth_factor, self.default_batch_size)
                           
    async def generate(self, job_input: JobInput):
        logger.info("[%s] vLLM generate starting (stream=%s, apply_chat_template=%s)", job_input.request_id, job_input.stream, job_input.apply_chat_template)
        try:
            async for batch in self._generate_vllm(
                llm_input=job_input.llm_input,
                validated_sampling_params=job_input.sampling_params,
                batch_size=job_input.max_batch_size,
                stream=job_input.stream,
                apply_chat_template=job_input.apply_chat_template,
                request_id=job_input.request_id,
                batch_size_growth_factor=job_input.batch_size_growth_factor,
                min_batch_size=job_input.min_batch_size
            ):
                yield batch
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[%s] vLLM generate failed: %s\n%s", job_input.request_id, e, tb)
            yield {"error": {"message": f"{type(e).__name__}: {e}", "traceback": tb}}

    async def _generate_vllm(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id, batch_size_growth_factor, min_batch_size: str) -> AsyncGenerator[dict, None]:
        try:
            if apply_chat_template or isinstance(llm_input, list):
                logger.info("[%s] Applying chat template to input", request_id)
                tokenizer_wrapper = self._get_tokenizer_for_chat_template()
                llm_input = tokenizer_wrapper.apply_chat_template(llm_input)

            prompt_preview = (llm_input[:120] + "...") if isinstance(llm_input, str) and len(llm_input) > 120 else llm_input
            logger.info("[%s] Submitting prompt to vLLM (n=%d, max_tokens=%s, prompt_preview=%s)", request_id, validated_sampling_params.n, validated_sampling_params.max_tokens, repr(prompt_preview))

            gen_start = time.time()
            results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
            n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
            last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

            batch = {
                "choices": [{"tokens": []} for _ in range(n_responses)],
            }
            
            max_batch_size = batch_size or self.default_batch_size
            batch_size_growth_factor, min_batch_size = batch_size_growth_factor or self.batch_size_growth_factor, min_batch_size or self.min_batch_size
            batch_size = BatchSize(max_batch_size, min_batch_size, batch_size_growth_factor)
            batches_yielded = 0

            async for request_output in results_generator:
                if is_first_output:  # Count input tokens only once
                    n_input_tokens = len(request_output.prompt_token_ids)
                    first_token_time = time.time() - gen_start
                    logger.info("[%s] First token received in %.2fs (input_tokens=%d)", request_id, first_token_time, n_input_tokens)
                    is_first_output = False

                for output in request_output.outputs:
                    output_index = output.index
                    token_counters["total"] += 1
                    if stream:
                        new_output = output.text[len(last_output_texts[output_index]):]
                        batch["choices"][output_index]["tokens"].append(new_output)
                        token_counters["batch"] += 1

                        if token_counters["batch"] >= batch_size.current_batch_size:
                            batch["usage"] = {
                                "input": n_input_tokens,
                                "output": token_counters["total"],
                            }
                            yield batch
                            batches_yielded += 1
                            batch = {
                                "choices": [{"tokens": []} for _ in range(n_responses)],
                            }
                            token_counters["batch"] = 0
                            batch_size.update()

                    last_output_texts[output_index] = output.text

            if not stream:
                for output_index, output in enumerate(last_output_texts):
                    batch["choices"][output_index]["tokens"] = [output]
                token_counters["batch"] += 1

            if token_counters["batch"] > 0:
                batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
                yield batch
                batches_yielded += 1

            gen_duration = time.time() - gen_start
            tokens_per_sec = token_counters["total"] / gen_duration if gen_duration > 0 else 0
            logger.info("[%s] Generation complete in %.2fs — input_tokens=%d, output_tokens=%d, %.1f tok/s, batches=%d",
                         request_id, gen_duration, n_input_tokens, token_counters["total"], tokens_per_sec, batches_yielded)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[%s] Error during vLLM generation: %s\n%s", request_id, e, tb)
            yield {"error": {"message": f"Generation failed — {type(e).__name__}: {e}", "traceback": tb}}

    def _initialize_llm(self):
        try:
            start = time.time()
            engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            end = time.time()
            logging.info(f"Initialized vLLM engine in {end - start:.2f}s")
            return engine
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e


class OpenAIvLLMEngine(vLLMEngine):
    def __init__(self, vllm_engine):
        super().__init__(vllm_engine)
        logger.info("Initializing OpenAI-compatible engine...")
        self.served_model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or self.engine_args.model
        self.response_role = os.getenv("OPENAI_RESPONSE_ROLE") or "assistant"
        self.lora_adapters = self._load_lora_adapters()
        asyncio.run(self._initialize_engines())
        # Handle both integer and boolean string values for RAW_OPENAI_OUTPUT
        raw_output_env = os.getenv("RAW_OPENAI_OUTPUT", "1")
        if raw_output_env.lower() in ('true', 'false'):
            self.raw_openai_output = raw_output_env.lower() == 'true'
        else:
            self.raw_openai_output = bool(int(raw_output_env))
        logger.info("OpenAI engine ready (served_model=%s, raw_output=%s, lora_adapters=%d)",
                     self.served_model_name, self.raw_openai_output, len(self.lora_adapters))

    def _load_lora_adapters(self):
        adapters = []
        try:
            adapters = json.loads(os.getenv("LORA_MODULES", '[]'))
        except Exception as e:
            logging.info(f"---Initialized adapter json load error: {e}")

        for i, adapter in enumerate(adapters):
            try:
                adapters[i] = LoRAModulePath(**adapter)
                logging.info(f"---Initialized adapter: {adapter}")
            except Exception as e:
                logging.info(f"---Initialized adapter not worked: {e}")
                continue
        return adapters

    async def _initialize_engines(self):
        self.model_config = self.llm.model_config
        self.base_model_paths = [
            BaseModelPath(name=self.engine_args.model, model_path=self.engine_args.model)
        ]

        self.serving_models = OpenAIServingModels(
            engine_client=self.llm,
            base_model_paths=self.base_model_paths,
            lora_modules=self.lora_adapters,
        )
        await self.serving_models.init_static_loras()
        
        # Get chat template from vLLM tokenizer if available
        chat_template = None
        if self.tokenizer and hasattr(self.tokenizer, 'tokenizer'):
            chat_template = self.tokenizer.tokenizer.chat_template
        
        self.chat_engine = OpenAIServingChat(
            engine_client=self.llm,
            models=self.serving_models,
            response_role=self.response_role,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            # enable_reasoning=os.getenv('ENABLE_REASONING', 'false').lower() == 'true',
            reasoning_parser=os.getenv('REASONING_PARSER', ""),
            # return_token_as_token_ids=False,
            enable_auto_tools=os.getenv('ENABLE_AUTO_TOOL_CHOICE', 'false').lower() == 'true',
            tool_parser=os.getenv('TOOL_CALL_PARSER', "") or None,
            enable_prompt_tokens_details=False
        )
        self.completion_engine = OpenAIServingCompletion(
            engine_client=self.llm,
            models=self.serving_models,
            request_logger=None,
            # return_token_as_token_ids=False,
        )
    
    async def generate(self, openai_request: JobInput):
        logger.info("[OpenAI] Handling route: %s", openai_request.openai_route)
        try:
            if openai_request.openai_route == "/v1/models":
                logger.info("[OpenAI] Serving model list request")
                yield await self._handle_model_request()
            elif openai_request.openai_route in ["/v1/chat/completions", "/v1/completions"]:
                async for response in self._handle_chat_or_completion_request(openai_request):
                    yield response
            else:
                logger.warning("[OpenAI] Invalid route requested: %s", openai_request.openai_route)
                yield create_error_response("Invalid route").model_dump()
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[OpenAI] Unhandled error in generate: %s\n%s", e, tb)
            yield {"error": {"message": f"{type(e).__name__}: {e}", "traceback": tb}}
    
    async def _handle_model_request(self):
        models = await self.serving_models.show_available_models()
        return models.model_dump()
    
    def _is_qwen_omni_model(self):
        """Check if the current model is a Qwen3-Omni variant."""
        model_name = (self.engine_args.model or "").lower()
        if os.getenv("ENABLE_QWEN_OMNI_AUDIO", "").lower() == "true":
            return True
        return "qwen3-omni" in model_name

    async def _handle_chat_or_completion_request(self, openai_request: JobInput):
        stream = openai_request.openai_input.get("stream", False)
        model_requested = openai_request.openai_input.get("model", "default")

        # Route audio+text requests through custom Qwen-Omni preprocessing
        if (openai_request.openai_route == "/v1/chat/completions"
                and self._is_qwen_omni_model()):
            from multimodal_processor import has_audio_content
            if has_audio_content(openai_request.openai_input.get("messages", [])):
                logger.info("[OpenAI] Audio content detected — routing to Qwen-Omni audio handler")
                async for response in self._handle_audio_chat_request(openai_request):
                    yield response
                return

        if openai_request.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
            generator_function = self.chat_engine.create_chat_completion
            logger.info("[OpenAI] Processing chat completion (model=%s, stream=%s)", model_requested, stream)
        elif openai_request.openai_route == "/v1/completions":
            request_class = CompletionRequest
            generator_function = self.completion_engine.create_completion
            logger.info("[OpenAI] Processing text completion (model=%s, stream=%s)", model_requested, stream)
        
        try:
            request = request_class(
                **openai_request.openai_input
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[OpenAI] Failed to parse request: %s\n%s", e, tb)
            yield {"error": {"message": f"Request parsing failed — {type(e).__name__}: {e}", "traceback": tb}}
            return
        
        try:
            dummy_request = DummyRequest()
            gen_start = time.time()
            response_generator = await generator_function(request, raw_request=dummy_request)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[OpenAI] Failed to start generation: %s\n%s", e, tb)
            yield {"error": {"message": f"Generation start failed — {type(e).__name__}: {e}", "traceback": tb}}
            return

        try:
            if not stream or isinstance(response_generator, ErrorResponse):
                if isinstance(response_generator, ErrorResponse):
                    logger.warning("[OpenAI] Engine returned an error response")
                else:
                    logger.info("[OpenAI] Non-streaming response generated in %.2fs", time.time() - gen_start)
                yield response_generator.model_dump()
            else:
                batch = []
                batch_token_counter = 0
                total_chunks = 0
                batch_size = BatchSize(self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor)
            
                async for chunk_str in response_generator:
                    if "data" in chunk_str:
                        if self.raw_openai_output:
                            data = chunk_str
                        elif "[DONE]" in chunk_str:
                            continue
                        else:
                            data = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) if not self.raw_openai_output else chunk_str
                        batch.append(data)
                        batch_token_counter += 1
                        total_chunks += 1
                        if batch_token_counter >= batch_size.current_batch_size:
                            if self.raw_openai_output:
                                batch = "".join(batch)
                            yield batch
                            batch = []
                            batch_token_counter = 0
                            batch_size.update()
                if batch:
                    if self.raw_openai_output:
                        batch = "".join(batch)
                    yield batch

                logger.info("[OpenAI] Streaming complete in %.2fs — %d chunks sent", time.time() - gen_start, total_chunks)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[OpenAI] Error during response streaming: %s\n%s", e, tb)
            yield {"error": {"message": f"Streaming failed — {type(e).__name__}: {e}", "traceback": tb}}

    async def _handle_audio_chat_request(self, openai_request: JobInput):
        """Handle chat completion with audio content using Qwen-Omni preprocessing."""
        from multimodal_processor import build_vllm_inputs, get_processor

        openai_input = openai_request.openai_input
        messages = openai_input["messages"]
        stream = openai_input.get("stream", False)
        n_messages = len(messages)

        logger.info("[Audio] Starting audio chat request (messages=%d, stream=%s)", n_messages, stream)

        try:
            logger.info("[Audio] Loading processor for model: %s", self.engine_args.model)
            processor = get_processor(self.engine_args.model)
            logger.info("[Audio] Building vLLM inputs from %d message(s)...", n_messages)
            inputs, limit_mm = build_vllm_inputs(messages, processor)
            logger.info("[Audio] Inputs built successfully (audio_limit=%s, prompt_length=%d chars)",
                         limit_mm, len(inputs.get("prompt", "")))
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"{type(e).__name__}: {e}" if str(e).strip() else type(e).__name__
            logger.error("[Audio] Failed to build audio inputs: %s\n%s", err_msg, tb)
            yield {"error": {"message": f"Audio preprocessing failed: {err_msg}", "traceback": tb}}
            return

        # Build sampling params from request
        sp_kwargs = {}
        max_tokens = openai_input.get("max_completion_tokens") or openai_input.get("max_tokens")
        if max_tokens is not None:
            sp_kwargs["max_tokens"] = max_tokens
        if openai_input.get("temperature") is not None:
            sp_kwargs["temperature"] = openai_input["temperature"]
        if openai_input.get("top_p") is not None:
            sp_kwargs["top_p"] = openai_input["top_p"]
        if openai_input.get("n") is not None:
            sp_kwargs["n"] = openai_input["n"]
        if openai_input.get("stop") is not None:
            sp_kwargs["stop"] = openai_input["stop"]
        if openai_input.get("presence_penalty") is not None:
            sp_kwargs["presence_penalty"] = openai_input["presence_penalty"]
        if openai_input.get("frequency_penalty") is not None:
            sp_kwargs["frequency_penalty"] = openai_input["frequency_penalty"]
        if openai_input.get("seed") is not None:
            sp_kwargs["seed"] = openai_input["seed"]

        # Handle response_format -> structured outputs
        response_format = openai_input.get("response_format")
        if response_format:
            logger.info("[Audio] Response format requested: %s", response_format.get("type"))
            try:
                from vllm.sampling_params import StructuredOutputsParams
                rf_type = response_format.get("type")
                if rf_type == "json_schema":
                    schema = response_format.get("json_schema", {}).get("schema")
                    if schema:
                        sp_kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
                elif rf_type == "json_object":
                    sp_kwargs["structured_outputs"] = StructuredOutputsParams(json={})
            except ImportError:
                logger.warning("[Audio] StructuredOutputsParams not available, response_format ignored")

        sampling_params = SamplingParams(**sp_kwargs)
        request_id = random_uuid()
        created_time = int(time.time())
        logger.info("[Audio] Sampling params: max_tokens=%s, temperature=%s — request_id=%s",
                     sp_kwargs.get("max_tokens"), sp_kwargs.get("temperature"), request_id)

        try:
            logger.info("[Audio] Submitting audio generation to vLLM...")
            gen_start = time.time()
            results_generator = self.llm.generate(inputs, sampling_params, request_id)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[Audio] Failed to start audio generation: %s\n%s", e, tb)
            yield {"error": {"message": f"Audio generation failed to start: {type(e).__name__}: {e}", "traceback": tb}}
            return

        try:
            if not stream:
                # Non-streaming: collect all output then return full response
                logger.info("[Audio] Collecting non-streaming output...")
                final_output = None
                async for request_output in results_generator:
                    final_output = request_output

                if final_output is None:
                    logger.warning("[Audio] No output generated from vLLM")
                    yield {"error": {"message": "No output generated from audio model"}}
                    return

                choices = []
                for output in final_output.outputs:
                    choices.append({
                        "index": output.index,
                        "message": {
                            "role": "assistant",
                            "content": output.text,
                        },
                        "finish_reason": output.finish_reason or "stop",
                        "logprobs": None,
                    })

                n_prompt_tokens = len(final_output.prompt_token_ids)
                n_completion_tokens = sum(len(o.token_ids) for o in final_output.outputs)
                audio_duration = time.time() - gen_start

                logger.info("[Audio] Non-streaming response ready in %.2fs — prompt_tokens=%d, completion_tokens=%d",
                             audio_duration, n_prompt_tokens, n_completion_tokens)

                yield {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": created_time,
                    "model": self.served_model_name,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": n_prompt_tokens,
                        "completion_tokens": n_completion_tokens,
                        "total_tokens": n_prompt_tokens + n_completion_tokens,
                    },
                }
            else:
                # Streaming: yield batched chunks
                logger.info("[Audio] Starting streaming output...")
                n_responses = openai_input.get("n", 1)
                last_output_texts = ["" for _ in range(n_responses)]
                batch = []
                batch_token_counter = 0
                total_chunks = 0
                batch_size = BatchSize(self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor)

                async for request_output in results_generator:
                    for output in request_output.outputs:
                        new_text = output.text[len(last_output_texts[output.index]):]
                        if new_text:
                            chunk = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": self.served_model_name,
                                "choices": [{
                                    "index": output.index,
                                    "delta": {"content": new_text},
                                    "finish_reason": None,
                                }],
                            }
                            if self.raw_openai_output:
                                batch.append(f"data: {json.dumps(chunk)}\n\n")
                            else:
                                batch.append(chunk)
                            batch_token_counter += 1
                            total_chunks += 1

                            if batch_token_counter >= batch_size.current_batch_size:
                                if self.raw_openai_output:
                                    yield "".join(batch)
                                else:
                                    yield batch
                                batch = []
                                batch_token_counter = 0
                                batch_size.update()

                        last_output_texts[output.index] = output.text

                # Emit finish chunks
                for i in range(n_responses):
                    finish_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": self.served_model_name,
                        "choices": [{
                            "index": i,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    if self.raw_openai_output:
                        batch.append(f"data: {json.dumps(finish_chunk)}\n\n")
                    else:
                        batch.append(finish_chunk)
                    batch_token_counter += 1

                if self.raw_openai_output:
                    batch.append("data: [DONE]\n\n")

                if batch:
                    if self.raw_openai_output:
                        yield "".join(batch)
                    else:
                        yield batch

                audio_duration = time.time() - gen_start
                logger.info("[Audio] Streaming complete in %.2fs — %d chunks sent", audio_duration, total_chunks)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[Audio] Error during audio generation output: %s\n%s", e, tb)
            yield {"error": {"message": f"Audio generation failed — {type(e).__name__}: {e}", "traceback": tb}}
