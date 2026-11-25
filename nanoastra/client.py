"""Chat client for an OpenAI-compatible server (vLLM)."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Sent with every request: the sampling of all nanoAstra results, thinking switched off.
REQUEST = {
    "max_tokens": 2048,
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "repetition_penalty": 1.0, "presence_penalty": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
STREAM = {"stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
JPEG_QUALITY = 95


def encode_image(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def image_part(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


@dataclass
class Reply:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks: list = field(default_factory=list)      # (characters, output tokens) received so far, per stream chunk
    seconds: float = 0.0
    error: Optional[str] = None


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout_s: float = 600.0,
                 attempts: int = 4) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout_s = timeout_s
        self.attempts = attempts

    def body(self, messages: list[dict], extra: Optional[dict] = None) -> dict:
        return {"model": self.model, "messages": messages, **REQUEST, **(extra or {})}

    def chat(self, messages: list[dict], extra: Optional[dict] = None) -> Reply:
        """One reply, streamed.  A failed request is retried; after the last attempt the error is returned."""
        data = json.dumps({**self.body(messages, extra), **STREAM}).encode()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        started, error = time.time(), None
        for attempt in range(1, self.attempts + 1):
            try:
                request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    reply = _read_stream(response)
                reply.seconds = time.time() - started
                return reply
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}"
                if exc.code in (400, 401, 403, 404, 413, 422):
                    break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            if attempt < self.attempts:
                logger.warning("request failed (%s); retrying", error[:160])
                time.sleep(min(30.0, 2.0 ** attempt))
        return Reply(seconds=time.time() - started, error=error)


def _read_stream(response) -> Reply:
    reply, pieces = Reply(), []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        if event.get("error"):
            raise RuntimeError(f"stream error: {event['error']}")
        usage = event.get("usage") or {}
        if usage:
            reply.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            reply.completion_tokens = int(usage.get("completion_tokens") or 0)
        for choice in event.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content") or ""
            if piece:
                pieces.append(piece)
                tokens = usage.get("completion_tokens")
                reply.chunks.append((sum(map(len, pieces)), len(reply.chunks) + 1 if tokens is None else tokens))
    reply.text = "".join(pieces)
    return reply
