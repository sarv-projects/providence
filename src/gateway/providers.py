"""
OpenAI-compatible provider client (stdlib only).

Groq, OpenAI, OpenRouter, Together, Azure (with OpenAI-compatible endpoint),
local vLLM/Ollama servers, etc. all expose the ``/chat/completions`` REST shape.
This tiny client talks to any of them over HTTPS using only ``urllib`` so the
gateway needs zero extra dependencies. Streaming is intentionally left for a
future iteration (the router currently returns the full completion); the class
is structured so a streaming transport can slot in.

Raises:
    ProviderHTTPError     — JSON error body with an HTTP status
    ProviderTimeoutError  — request exceeded timeout
    ProviderConnectionError — network-level failure (DNS/TLS/connection refused)
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List

DEFAULT_TIMEOUT = 180.0

# Per-model pricing in USD per 1M tokens for cost accounting. Tune as needed.
# (input, output) pairs.
PRICING: Dict[str, tuple] = {
    # Groq production ids often use openai/ prefix; keep bare + prefixed aliases.
    "gpt-oss-20b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gpt-oss-120b": (0.30, 1.20),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
    # Free models (cost accounting only; actual cost = 0)
    "mimo-v2.5-free": (0.0, 0.0),
    "deepseek-v4-flash-free": (0.0, 0.0),
    "big-pickle": (0.0, 0.0),
    "nemotron-3-ultra-free": (0.0, 0.0),
    "nemotron-3.5-lightning-free": (0.0, 0.0),
    "hy3-free": (0.0, 0.0),
    "laguna-s-2.1-free": (0.0, 0.0),
    # Gemini free tier (free input/output on eligible models)
    "gemini-3.6-flash": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "*default": (0.50, 1.50),
}


class ProviderHTTPError(Exception):
    def __init__(self, status: int, message: str, retriable: bool) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retriable = retriable


class ProviderTimeoutError(Exception):
    pass


class ProviderConnectionError(Exception):
    pass


@dataclass
class ProviderResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0


def retriable_status(status: int) -> bool:
    """Which upstream statuses should trigger retry/failover (vs. client bugs)."""
    return status in (408, 429, 500, 502, 503, 504)


def _extract_stream_text(chunk: dict) -> str:
    """Pull streamed text out of any common provider SSE payload shape.

    Returns "" for frames that carry no content (usage, pings, errors,
    message_start, tool calls, etc.). Previously this was hard-wired to the
    OpenAI ``choices[0].delta.content`` shape only.
    """
    # OpenAI-compatible (Groq, OpenAI, OpenRouter, vLLM, Ollama, …)
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0] if isinstance(choices[0], dict) else {}
        content = (
            delta.get("delta", {}).get("content")
            or delta.get("text")              # completion-style streams
            or delta.get("message", {}).get("content")
            or ""
        )
        return content or ""

    # Anthropic Messages API
    if chunk.get("type") == "content_block_delta":
        d = chunk.get("delta") or {}
        return d.get("text") or d.get("thinking") or ""

    # Cohere v2 / simple text frames
    if isinstance(chunk.get("text"), str):
        return chunk["text"]
    if isinstance(chunk.get("delta"), str):
        return chunk["delta"]

    return ""


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        protocol: str = "openai_chat",
    ) -> None:
        self.name = name
        self.protocol = protocol or "openai_chat"
        base_url = base_url.rstrip("/")
        # OpenAI-compatible paths usually end with /v1; Anthropic uses /v1/messages
        if self.protocol == "openai_chat" and not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        self.base_url = base_url
        self.api_keys = [api_key] if api_key else []

    def _url(self) -> str:
        if self.protocol == "anthropic_messages":
            # https://api.anthropic.com/v1/messages
            base = self.base_url
            if base.endswith("/v1"):
                return f"{base}/messages"
            return f"{base}/v1/messages"
        if self.protocol == "cohere_v2_chat":
            base = self.base_url.rstrip("/")
            if base.endswith("/v2"):
                return f"{base}/chat"
            return f"{base}/v2/chat"
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ProviderResult:
        """Synchronous completion — returns the full response."""
        key = api_key or (self.api_keys[0] if self.api_keys else "")
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "AutonomousResearchAgent/1.0 (python-urllib)",
        }

        if self.protocol == "anthropic_messages":
            system = ""
            anthro_msgs = []
            for m in messages:
                if m.get("role") == "system":
                    system = (system + "\n" + m.get("content", "")).strip()
                else:
                    anthro_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            if not anthro_msgs:
                anthro_msgs = [{"role": "user", "content": ""}]
            payload: Dict = {
                "model": model,
                "messages": anthro_msgs,
                "max_tokens": max_tokens or 4096,
                "temperature": temperature,
            }
            if system:
                payload["system"] = system
            if key:
                headers["x-api-key"] = key
                headers["anthropic-version"] = "2023-06-01"
        elif self.protocol == "cohere_v2_chat":
            # Cohere v2 chat API
            system = next((m["content"] for m in messages if m.get("role") == "system"), "")
            chat_history = []
            user_msg = ""
            for m in messages:
                if m.get("role") == "system":
                    continue
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                elif m.get("role") == "assistant":
                    chat_history.append({"role": "assistant", "content": m.get("content", "")})
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": temperature,
            }
            if system:
                payload["messages"] = [{"role": "system", "content": system}] + payload["messages"]
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if key:
                headers["Authorization"] = f"Bearer {key}"
        else:
            payload = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if key:
                headers["Authorization"] = f"Bearer {key}"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(),
            data=body,
            method="POST",
            headers=headers,
        )
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            msg = ""
            try:
                body_err = json.loads(e.read().decode("utf-8", "ignore"))
                msg = (
                    body_err.get("error", {}).get("message")
                    or body_err.get("message")
                    or str(e)
                )
            except Exception:
                msg = str(e)
            raise ProviderHTTPError(e.code, msg, retriable_status(e.code))
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            if isinstance(e, (socket.timeout, TimeoutError)) or (
                isinstance(e, urllib.error.URLError) and isinstance(e.reason, (socket.timeout, TimeoutError))
            ):
                raise ProviderTimeoutError(f"{self.name} timed out: {e}")
            raise ProviderConnectionError(f"{self.name} connection error: {e}")

        latency = time.time() - start
        try:
            data = json.loads(raw.decode("utf-8"))
            if self.protocol == "anthropic_messages":
                blocks = data.get("content") or []
                text = "".join(
                    b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
                )
                usage = data.get("usage") or {}
                return ProviderResult(
                    text=text,
                    prompt_tokens=int(usage.get("input_tokens", 0)),
                    completion_tokens=int(usage.get("output_tokens", 0)),
                    model=data.get("model", model),
                    latency_s=latency,
                )
            if self.protocol == "cohere_v2_chat":
                # message.content can be list of {type,text} or string
                msg = data.get("message") or {}
                content = msg.get("content") or data.get("text") or ""
                if isinstance(content, list):
                    text = "".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                else:
                    text = str(content)
                usage = data.get("usage") or {}
                bil = usage.get("billed_units") or {}
                return ProviderResult(
                    text=text,
                    prompt_tokens=int(bil.get("input_tokens", 0) or usage.get("tokens", {}).get("input_tokens", 0) or 0),
                    completion_tokens=int(bil.get("output_tokens", 0) or usage.get("tokens", {}).get("output_tokens", 0) or 0),
                    model=data.get("model", model) or model,
                    latency_s=latency,
                )
            # OpenAI-compatible (incl. Zen free reasoning models)
            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            text = msg.get("content") or choice0.get("text") or ""
            if not text:
                # Reasoning models (big-pickle, deepseek-free, etc.) often put
                # output in reasoning_content / reasoning with empty content.
                text = (
                    msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or choice0.get("reasoning_content")
                    or ""
                )
            if text is None:
                text = ""
            usage = data.get("usage", {})
            return ProviderResult(
                text=str(text),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                model=data.get("model", model),
                latency_s=latency,
            )
        except (KeyError, ValueError, IndexError) as e:
            raise ProviderConnectionError(f"{self.name} bad response: {e}")

    def complete_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Streaming completion — yields text chunks via SSE-style generator.

        Yields str chunks as they arrive from the provider. Uses HTTP
        streaming response with stream=True in the payload.
        """
        key = api_key or (self.api_keys[0] if self.api_keys else "")
        payload: Dict = {
            "model": model, "messages": messages, "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        body = json.dumps(payload).encode("utf-8")
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "AutonomousResearchAgent/1.0 (python-urllib)",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            self._url(), data=body, method="POST", headers=headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8", "ignore").strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str.startswith("event:"):
                        continue  # Anthropic-style event framing — data lines carry the payload
                    if line_str == "data: [DONE]":
                        break
                    if line_str.startswith("data: "):
                        try:
                            chunk = json.loads(line_str[6:])
                        except json.JSONDecodeError:
                            continue
                        # Extract text tolerantly across provider SSE shapes:
                        #   OpenAI-compatible: {"choices":[{"delta":{"content": "..."}}]}
                        #   Anthropic:         {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
                        #   Cohere:            {"text": "..."} / {"delta": "..."}
                        #   Usage/error frames yield nothing and are skipped.
                        delta = _extract_stream_text(chunk)
                        if delta:
                            yield delta
        except urllib.error.HTTPError as e:
            msg = ""
            try:
                msg = json.loads(e.read().decode("utf-8", "ignore")).get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            raise ProviderHTTPError(e.code, msg, retriable_status(e.code))
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            if isinstance(e, (socket.timeout, TimeoutError)) or (
                isinstance(e, urllib.error.URLError) and isinstance(e.reason, (socket.timeout, TimeoutError))
            ):
                raise ProviderTimeoutError(f"{self.name} timed out: {e}")
            raise ProviderConnectionError(f"{self.name} connection error: {e}")
