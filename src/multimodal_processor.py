"""
Multimodal preprocessing for Qwen3-Omni audio+text inference.

Converts OpenAI-style messages with audio_url/audio_file content to the format
expected by qwen_omni_utils and Qwen3OmniMoeProcessor, then builds vLLM inputs.
"""

import base64
import io
import logging
from pathlib import Path
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def decode_audio_data_uri(data_uri: str) -> "tuple[np.ndarray, float] | None":
    """
    Decode data:audio/*;base64,... to (audio_array, sample_rate).

    qwen_omni_utils does not support base64 audio; vLLM expects (numpy_array, sample_rate).
    Returns None if the input is not a valid base64 audio data URI or decoding fails.
    """
    if not isinstance(data_uri, str):
        return None
    if not data_uri.startswith("data:audio/") or ";base64," not in data_uri:
        return None

    try:
        payload = data_uri.split(";base64,", 1)[1]
        audio_bytes = base64.b64decode(payload)
    except (IndexError, ValueError) as e:
        logger.warning("Failed to decode base64 audio data URI: %s", e)
        return None

    try:
        import librosa
        import numpy as np

        audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
        return (np.asarray(audio_array, dtype=np.float32), float(sample_rate))
    except Exception as e:
        logger.warning("Failed to load audio from base64 (ensure librosa/audioread available): %s", e)
        return None


def _extract_audio_url(item):
    """Extract data URI or URL from various process_mm_info return formats."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("url") or item.get("audio") or item.get("data")
    return None


def _convert_audios_for_vllm(audios):
    """
    Convert audios from process_mm_info to vLLM-compatible format.

    Decodes any base64 data URIs to (numpy_array, sample_rate) tuples.
    qwen_omni_utils does not support base64 audio; vLLM expects URLs or numpy tuples.
    Handles both bare strings and dict returns from process_mm_info.
    """
    if audios is None:
        return None

    def convert_one(item):
        url = _extract_audio_url(item)
        if url and isinstance(url, str) and url.startswith("data:audio/") and ";base64," in url:
            decoded = decode_audio_data_uri(url)
            if decoded is not None:
                return decoded
            # Decode failed; keep original (may fail downstream, but preserves behavior)
        return item

    if isinstance(audios, list):
        converted = [convert_one(a) for a in audios]
    else:
        converted = convert_one(audios)

    return converted


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

    # Decode base64 data URIs to (numpy_array, sample_rate); qwen_omni_utils does not support them
    audios = _convert_audios_for_vllm(audios)

    # Diagnostic logging (can remove after verification)
    if audios is not None:
        if isinstance(audios, list):
            logger.info(
                "build_vllm_inputs: audios type=list len=%d, first_item_type=%s",
                len(audios),
                type(audios[0]).__name__ if audios else "N/A",
            )
        else:
            logger.info("build_vllm_inputs: audios type=%s", type(audios).__name__)

    inputs = {
        "prompt": text,
        "multi_modal_data": {},
    }

    limit_mm_per_prompt = {}

    if audios is not None:
        # vLLM Qwen3-Omni: single audio = (array, sr), multiple = [(array, sr), ...]
        if isinstance(audios, list) and len(audios) == 1:
            inputs["multi_modal_data"]["audio"] = audios[0]
            n_audio = 1
        else:
            inputs["multi_modal_data"]["audio"] = audios
            n_audio = len(audios) if isinstance(audios, list) else 1
        limit_mm_per_prompt["audio"] = n_audio

    return inputs, limit_mm_per_prompt
