import os
import json
import logging
from torch.cuda import device_count
from vllm import AsyncEngineArgs
from src.utils import convert_limit_mm_per_prompt

RENAME_ARGS_MAP = {
    "MODEL_NAME": "model",
    "MODEL_REVISION": "revision",
    "TOKENIZER_NAME": "tokenizer",
}

DEFAULT_ARGS = {
    "disable_log_stats": os.getenv('DISABLE_LOG_STATS', 'False').lower() == 'true',
    "gpu_memory_utilization": float(os.getenv('GPU_MEMORY_UTILIZATION', 0.95)),
    "pipeline_parallel_size": int(os.getenv('PIPELINE_PARALLEL_SIZE', 1)),
    "tensor_parallel_size": int(os.getenv('TENSOR_PARALLEL_SIZE', 1)),
    "served_model_name": os.getenv('SERVED_MODEL_NAME', None),
    "tokenizer": os.getenv('TOKENIZER', None),
    "skip_tokenizer_init": os.getenv('SKIP_TOKENIZER_INIT', 'False').lower() == 'true',
    "tokenizer_mode": os.getenv('TOKENIZER_MODE', 'auto'),
    "trust_remote_code": os.getenv('TRUST_REMOTE_CODE', 'False').lower() == 'true',
    "download_dir": os.getenv('DOWNLOAD_DIR', None),
    "load_format": os.getenv('LOAD_FORMAT', 'auto'),
    "config_format": os.getenv('CONFIG_FORMAT', 'auto'),
    "dtype": os.getenv('DTYPE', 'auto'),
    "kv_cache_dtype": os.getenv('KV_CACHE_DTYPE', 'auto'),
    "seed": int(os.getenv('SEED', 0)),
    "max_model_len": int(os.getenv('MAX_MODEL_LEN', 0)) or None,
    "distributed_executor_backend": os.getenv('DISTRIBUTED_EXECUTOR_BACKEND', None),
    "max_parallel_loading_workers": int(os.getenv('MAX_PARALLEL_LOADING_WORKERS', 0)) or None,
    "block_size": int(os.getenv('BLOCK_SIZE', 16)),
    "enable_prefix_caching": os.getenv('ENABLE_PREFIX_CACHING', 'False').lower() == 'true',
    "disable_sliding_window": os.getenv('DISABLE_SLIDING_WINDOW', 'False').lower() == 'true',
    "swap_space": int(os.getenv('SWAP_SPACE', 4)),  # GiB
    "cpu_offload_gb": int(os.getenv('CPU_OFFLOAD_GB', 0)),  # GiB
    "max_num_batched_tokens": int(os.getenv('MAX_NUM_BATCHED_TOKENS', 0)) or None,
    "max_num_seqs": int(os.getenv('MAX_NUM_SEQS', 256)),
    "max_logprobs": int(os.getenv('MAX_LOGPROBS', 20)),  # Default value for OpenAI Chat Completions API
    "revision": os.getenv('REVISION', None),
    "code_revision": os.getenv('CODE_REVISION', None),
    "tokenizer_revision": os.getenv('TOKENIZER_REVISION', None),
    "quantization": os.getenv('QUANTIZATION', None),
    "enforce_eager": os.getenv('ENFORCE_EAGER', 'False').lower() == 'true',
    "disable_custom_all_reduce": os.getenv('DISABLE_CUSTOM_ALL_REDUCE', 'False').lower() == 'true',
    "enable_lora": os.getenv('ENABLE_LORA', 'False').lower() == 'true',
    "max_loras": int(os.getenv('MAX_LORAS', 1)),
    "max_lora_rank": int(os.getenv('MAX_LORA_RANK', 16)),
    "fully_sharded_loras": os.getenv('FULLY_SHARDED_LORAS', 'False').lower() == 'true',
    "lora_dtype": os.getenv('LORA_DTYPE', 'auto'),
    "max_cpu_loras": int(os.getenv('MAX_CPU_LORAS', 0)) or None,
    "ray_workers_use_nsight": os.getenv('RAY_WORKERS_USE_NSIGHT', 'False').lower() == 'true',
    "num_gpu_blocks_override": int(os.getenv('NUM_GPU_BLOCKS_OVERRIDE', 0)) or None,
    "model_loader_extra_config": os.getenv('MODEL_LOADER_EXTRA_CONFIG', None),
    "ignore_patterns": os.getenv('IGNORE_PATTERNS', None),
    "enable_chunked_prefill": os.getenv('ENABLE_CHUNKED_PREFILL', None),
    "enable_expert_parallel": bool(os.getenv('ENABLE_EXPERT_PARALLEL', 'False').lower() == 'true'),
    "otlp_traces_endpoint": os.getenv('OTLP_TRACES_ENDPOINT', None),
}
limit_mm_env = os.getenv('LIMIT_MM_PER_PROMPT')
if limit_mm_env is not None:
    DEFAULT_ARGS["limit_mm_per_prompt"] = convert_limit_mm_per_prompt(limit_mm_env)

def match_vllm_args(args):
    """Rename args to match vllm by:
    1. Renaming keys to lower case
    2. Renaming keys to match vllm
    3. Filtering args to match vllm's AsyncEngineArgs

    Args:
        args (dict): Dictionary of args

    Returns:
        dict: Dictionary of args with renamed keys
    """
    renamed_args = {RENAME_ARGS_MAP.get(k, k): v for k, v in args.items()}
    matched_args = {k: v for k, v in renamed_args.items() if k in AsyncEngineArgs.__dataclass_fields__}
    return {k: v for k, v in matched_args.items() if v not in [None, "", "None"]}
def get_local_args():
    """
    Retrieve local arguments from a JSON file.

    Returns:
        dict: Local arguments.
    """
    if not os.path.exists("/local_model_args.json"):
        return {}

    with open("/local_model_args.json", "r") as f:
        local_args = json.load(f)

    if local_args.get("MODEL_NAME") is None:
        logging.warning("Model name not found in /local_model_args.json. There maybe was a problem when baking the model in.")

    logging.info(f"Using baked in model with args: {local_args}")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    return local_args
def get_engine_args():
    # Start with default args
    args = DEFAULT_ARGS
    
    # Get env args that match keys in AsyncEngineArgs
    args.update(os.environ)
    
    # Get local args if model is baked in and overwrite env args
    args.update(get_local_args())
    
    # if args.get("TENSORIZER_URI"): TODO: add back once tensorizer is ready
    #     args["load_format"] = "tensorizer"
    #     args["model_loader_extra_config"] = TensorizerConfig(tensorizer_uri=args["TENSORIZER_URI"], num_readers=None)
    #     logging.info(f"Using tensorized model from {args['TENSORIZER_URI']}")
    
    
    # Rename and match to vllm args
    args = match_vllm_args(args)

    if args.get("load_format") == "bitsandbytes":
        args["quantization"] = args["load_format"]
    
    # Set tensor parallel size and max parallel loading workers if more than 1 GPU is available
    num_gpus = device_count()
    if num_gpus > 1:
        args["tensor_parallel_size"] = num_gpus
        args["max_parallel_loading_workers"] = None
        if os.getenv("MAX_PARALLEL_LOADING_WORKERS"):
            logging.warning("Overriding MAX_PARALLEL_LOADING_WORKERS with None because more than 1 GPU is available.")
    
    # Deprecated env args backwards compatibility
    if args.get("kv_cache_dtype") == "fp8_e5m2":
        args["kv_cache_dtype"] = "fp8"
        logging.warning("Using fp8_e5m2 is deprecated. Please use fp8 instead.")
    # if "gemma-2" in args.get("model", "").lower():
    #     os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    #     logging.info("Using FLASHINFER for gemma-2 model.")
        
    return AsyncEngineArgs(**args)
