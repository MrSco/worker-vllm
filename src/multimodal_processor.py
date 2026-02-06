"""
Multimodal preprocessing for Qwen3-Omni audio+text inference.

Converts OpenAI-style messages with audio_url/audio_file content to the format
expected by qwen_omni_utils and Qwen3OmniMoeProcessor, then builds vLLM inputs.
"""

import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


def has_audio_content(messages):
    """Check if any message contains audio content parts."""
    if not messages:
        return False
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") in ("audio_url", "audio_file", "audio"):
                return True
    return False


def normalize_messages_to_qwen_format(messages):
    """
    Convert OpenAI-style message content to qwen_omni_utils format.

    - type: "audio_url" with audio_url.url -> type: "audio" with audio: url
    - type: "audio_file" with audio_file.path -> type: "audio" with audio: file:///path
    - type: "text" stays as-is
    - type: "audio" (already qwen format) stays as-is
    - image/video types are skipped (text+audio only)
    """
    normalized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            normalized.append({"role": role, "content": content})
            continue

        new_parts = []
        for part in content:
            ptype = part.get("type", "")
            if ptype == "text":
                new_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "audio_url":
                url = part.get("audio_url", {}).get("url", "")
                if url:
                    new_parts.append({"type": "audio", "audio": url})
            elif ptype == "audio_file":
                path = part.get("audio_file", {}).get("path", "")
                if path:
                    if path.startswith(("file://", "http://", "https://")):
                        audio_ref = path
                    else:
                        p = Path(path).resolve()
                        audio_ref = f"file:///{p.as_posix()}"
                    new_parts.append({"type": "audio", "audio": audio_ref})
            elif ptype == "audio":
                new_parts.append(part)
            # Skip image/video - text+audio only

        if new_parts:
            normalized.append({"role": role, "content": new_parts})
        else:
            normalized.append({"role": role, "content": content})

    return normalized


@lru_cache(maxsize=1)
def get_processor(model_path):
    """Load and cache the Qwen3OmniMoeProcessor."""
    from transformers import Qwen3OmniMoeProcessor
    logger.info("Loading Qwen3OmniMoeProcessor from %s", model_path)
    return Qwen3OmniMoeProcessor.from_pretrained(model_path)


def build_vllm_inputs(messages, processor):
    """
    Build vLLM inputs from messages using processor and qwen_omni_utils.

    Returns:
        tuple: (inputs_dict, limit_mm_per_prompt)
            - inputs_dict: {"prompt": str, "multi_modal_data": {"audio": list}}
            - limit_mm_per_prompt: {"audio": int}
    """
    from qwen_omni_utils import process_mm_info

    qwen_messages = normalize_messages_to_qwen_format(messages)

    text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audios, images, videos = process_mm_info(
        qwen_messages,
        use_audio_in_video=False,
    )

    inputs = {
        "prompt": text,
        "multi_modal_data": {},
    }

    limit_mm_per_prompt = {}

    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios
        n_audio = len(audios) if isinstance(audios, list) else 1
        limit_mm_per_prompt["audio"] = n_audio

    return inputs, limit_mm_per_prompt
