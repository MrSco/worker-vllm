FROM vllm/vllm-openai:v0.15.1

# Remove CUDA compat library override that breaks Blackwell GPUs
# See: https://github.com/vllm-project/vllm/pull/33992
RUN rm -f /etc/ld.so.conf.d/cuda-compat.conf && ldconfig

# Install ffmpeg for audio processing (Qwen3-Omni)
RUN apt-get update -y \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM audio extras and additional Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade -r /requirements.txt && \
    pip install "vllm[audio]==0.15.1"

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"


COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Override vllm entrypoint and start the handler
ENTRYPOINT []
CMD ["python3", "/src/handler.py"]
