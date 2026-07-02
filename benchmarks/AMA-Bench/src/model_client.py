import json
import argparse
import sys
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import yaml
import threading
from utils.usage_logger import (
    UsageLogger,
    approx_tokens,
    current_usage_context,
    now_iso,
    usage_to_dict,
)

class ModelClient:
    """Unified client for different model providers."""

    def __init__(self, config_path: str, server_type: str = "api", **kwargs):
        """
        Initialize model client.

        Args:
            config_path: Path to YAML config file containing provider, model, api_key, etc.
            server_type: Server type ("api" or "vllm")
            **kwargs: Additional provider-specific arguments (e.g., host, port for vllm)
        """
        # Load config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Extract provider and model from config
        self.provider = config.get('provider', '').lower()
        self.model = config.get('model', '')
        self.config = config
        vllm_launch = config.get('vllm_launch', {})
        self._max_model_len = (
            config.get('max_model_len')
            or vllm_launch.get('max_model_len', 131072)
        )

        # For vllm server type, override provider
        if server_type == "vllm":
            self.provider = "custom"
            # Use host/port from kwargs or config (support both 'host' and 'vllm_host' naming)
            host = config.get('vllm_host') or config.get('host', 'localhost')
            port = config.get('vllm_port') or config.get('port', 8000)
            config["base_url"] = f"http://{host}:{port}/v1"

        self._timeout = float(config.get('timeout', 600.0))
        self._usage_logger = None
        self._usage_role = kwargs.get("usage_role", "llm")
        self._min_request_interval = float(config.get("min_request_interval", 0.0) or 0.0)
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self.client = self._initialize_client()

    def set_usage_log(self, path: str, role: str = "llm"):
        self._usage_logger = UsageLogger(path)
        self._usage_role = role

    def _record_usage(
        self,
        *,
        prompt: str,
        response_text: str = "",
        max_tokens: int,
        temperature: float,
        elapsed_seconds: float,
        attempt: int,
        status: str,
        usage: Any = None,
        finish_reason: Optional[str] = None,
        error: Optional[str] = None,
        api: str = "chat.completions",
    ) -> None:
        if self._usage_logger is None:
            return
        usage_dict = usage_to_dict(usage)
        ctx = current_usage_context()
        prompt_tokens = (
            usage_dict.get("prompt_tokens")
            or usage_dict.get("input_tokens")
        )
        completion_tokens = (
            usage_dict.get("completion_tokens")
            or usage_dict.get("output_tokens")
        )
        total_tokens = usage_dict.get("total_tokens")
        self._usage_logger.write({
            "timestamp": now_iso(),
            "role": self._usage_role,
            "provider": self.provider,
            "model": self.model,
            "api": api,
            "status": status,
            "attempt": attempt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "finish_reason": finish_reason,
            "prompt_chars": len(prompt),
            "response_chars": len(response_text or ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage": usage_dict,
            "prompt_tokens_est": approx_tokens(prompt),
            "completion_tokens_est": approx_tokens(response_text or ""),
            "stage": ctx.get("stage"),
            "episode_id": ctx.get("episode_id"),
            "qa_index": ctx.get("qa_index"),
            "context": ctx,
            "error": error,
        })

    def _initialize_client(self):
        """Initialize provider-specific client."""
        if self.provider == "custom":
            from openai import OpenAI
            base_url = (
                self.config.get("base_url")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("LOCA_OPENAI_BASE_URL")
            )
            api_key = (
                self.config.get("api_key")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("LOCA_OPENAI_API_KEY")
                or "EMPTY"
            )
            return OpenAI(base_url=base_url, api_key=api_key, timeout=self._timeout)

        elif self.provider == "openai":
            from openai import OpenAI
            # Use api_key from config if provided, otherwise will use OPENAI_API_KEY env var
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAI(api_key=api_key, timeout=self._timeout)
            else:
                # Let OpenAI SDK handle the API key (will use OPENAI_API_KEY env var)
                return OpenAI(timeout=self._timeout)

        elif self.provider == "deepseek":
            from openai import OpenAI
            api_key = self.config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key not found in config or DEEPSEEK_API_KEY environment variable")
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=self._timeout)

        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found in config or GOOGLE_API_KEY environment variable")
            genai.configure(api_key=api_key)
            return genai

        elif self.provider in ["anthropic", "claude"]:
            from anthropic import Anthropic
            api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                return Anthropic(api_key=api_key)
            else:
                # Let Anthropic SDK handle the API key (will use ANTHROPIC_API_KEY env var)
                return Anthropic()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def query(self, prompt: str, temperature: float = 0.0, max_tokens: int = 4096, max_retries: int = 3, system: Optional[str] = None) -> str:
        """Query model with prompt with retry logic for rate limits."""
        import re as _re
        _truncated = False
        attempt = 0
        max_retries = int(self.config.get("max_retries", max_retries))
        rate_limit_base_wait = float(self.config.get("rate_limit_base_wait", 1.0))
        rate_limit_max_wait = float(self.config.get("rate_limit_max_wait", 60.0))
        rate_limit_jitter = float(self.config.get("rate_limit_jitter", 0.25))
        reasoning_effort = self.config.get("reasoning_effort")
        extra_body = self.config.get("extra_body")
        while attempt < max_retries:
            if self._min_request_interval > 0:
                with self._request_lock:
                    now = time.perf_counter()
                    wait_for = self._min_request_interval - (now - self._last_request_at)
                    if wait_for > 0:
                        time.sleep(wait_for)
                    self._last_request_at = time.perf_counter()
            call_start = time.perf_counter()
            try:
                if self.provider in ["custom", "deepseek"]:
                    provider_kwargs = {}
                    if reasoning_effort:
                        provider_kwargs["reasoning_effort"] = reasoning_effort
                    if extra_body:
                        provider_kwargs["extra_body"] = extra_body
                    try:
                        if self.model.startswith("gpt-5"):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_completion_tokens=max_tokens,
                                **provider_kwargs,
                            )
                        else:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                                **provider_kwargs,
                            )
                    except Exception as e:
                        if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_completion_tokens=max_tokens,
                                **provider_kwargs,
                            )
                        elif "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                                **provider_kwargs,
                            )
                        else:
                            raise
                    message = response.choices[0].message
                    finish_reason = response.choices[0].finish_reason
                    if message.content is None:
                        usage = getattr(response, "usage", None)
                        self._record_usage(
                            prompt=prompt,
                            response_text="",
                            max_tokens=max_tokens,
                            temperature=temperature,
                            elapsed_seconds=time.perf_counter() - call_start,
                            attempt=attempt + 1,
                            status="empty",
                            usage=usage,
                            finish_reason=finish_reason,
                        )
                        raise RuntimeError(
                            f"Empty content from custom provider "
                            f"(finish_reason={finish_reason}, usage={usage})"
                        )
                    content = message.content.strip()
                    self._record_usage(
                        prompt=prompt,
                        response_text=content,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        elapsed_seconds=time.perf_counter() - call_start,
                        attempt=attempt + 1,
                        status="ok",
                        usage=getattr(response, "usage", None),
                        finish_reason=finish_reason,
                    )
                    return content

                elif self.provider == "openai":
                    # gpt-5 family: hidden chain-of-thought tokens count against
                    # max_completion_tokens, so the chat.completions path can
                    # emit empty visible output even with a generous budget.
                    # Use the Responses API with reasoning_effort=minimal, the
                    # same convention as simpleqa / aa-lcr / Harbor parity.
                    if self.model.startswith("gpt-5"):
                        reasoning_effort = self.config.get("reasoning_effort", "minimal")
                        response = self.client.responses.create(
                            model=self.model,
                            input=prompt,
                            max_output_tokens=max_tokens,
                            reasoning={"effort": reasoning_effort},
                        )
                        content = (response.output_text or "").strip()
                        self._record_usage(
                            prompt=prompt,
                            response_text=content,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            elapsed_seconds=time.perf_counter() - call_start,
                            attempt=attempt + 1,
                            status="ok",
                            usage=getattr(response, "usage", None),
                            finish_reason=None,
                            api="responses",
                        )
                        return content
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens,
                        )
                        content = response.choices[0].message.content.strip()
                        self._record_usage(
                            prompt=prompt,
                            response_text=content,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            elapsed_seconds=time.perf_counter() - call_start,
                            attempt=attempt + 1,
                            status="ok",
                            usage=getattr(response, "usage", None),
                            finish_reason=response.choices[0].finish_reason,
                        )
                        return content
                    except Exception as e:
                        if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            content = response.choices[0].message.content.strip()
                            self._record_usage(
                                prompt=prompt,
                                response_text=content,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                elapsed_seconds=time.perf_counter() - call_start,
                                attempt=attempt + 1,
                                status="ok",
                                usage=getattr(response, "usage", None),
                                finish_reason=response.choices[0].finish_reason,
                            )
                            return content
                        else:
                            raise

                elif self.provider == "gemini":
                    model = self.client.GenerativeModel(self.model)
                    response = model.generate_content(
                        prompt,
                        generation_config=self.client.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    content = response.text.strip()
                    self._record_usage(
                        prompt=prompt,
                        response_text=content,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        elapsed_seconds=time.perf_counter() - call_start,
                        attempt=attempt + 1,
                        status="ok",
                        usage=getattr(response, "usage_metadata", None),
                        finish_reason=None,
                        api="gemini.generate_content",
                    )
                    return content

                elif self.provider in ["anthropic", "claude"]:
                    request_params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if system:
                        request_params["system"] = system
                    response = self.client.messages.create(**request_params)
                    if not response.content:
                        if hasattr(response, 'stop_reason') and response.stop_reason == 'refusal':
                            raise ValueError(f"Claude refused to respond. This may be due to content policy. Stop reason: {response.stop_reason}")
                        raise ValueError(f"Empty response content from Claude API. Response: {response}")
                    if not hasattr(response.content[0], 'text'):
                        raise ValueError(f"Response content has no text attribute. Content type: {type(response.content[0])}, Content: {response.content[0]}")
                    content = response.content[0].text.strip()
                    self._record_usage(
                        prompt=prompt,
                        response_text=content,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        elapsed_seconds=time.perf_counter() - call_start,
                        attempt=attempt + 1,
                        status="ok",
                        usage=getattr(response, "usage", None),
                        finish_reason=getattr(response, "stop_reason", None),
                        api="messages",
                    )
                    return content

                else:
                    raise ValueError(f"Query not implemented for provider: {self.provider}")

            except Exception as e:
                error_str = str(e)
                self._record_usage(
                    prompt=prompt,
                    response_text="",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    elapsed_seconds=time.perf_counter() - call_start,
                    attempt=attempt + 1,
                    status="error",
                    error=error_str[:1000],
                )
                if self.provider == "custom" and "MALFORMED_FUNCTION_CALL" in error_str:
                    prompt = (
                        "Do not use tool calls or function calls. Return plain text only.\n\n"
                        + prompt
                    )
                    attempt += 1
                    if attempt >= max_retries:
                        raise
                    print(f"Malformed function call from proxy, retrying as plain text... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                    continue
                # Don't retry on refusals or permanent errors
                if "refused" in error_str.lower() or "refusal" in error_str.lower():
                    raise
                # Handle context-length 400 errors: reduce max_tokens without consuming an attempt
                is_context_length_error = (
                    ("400" in error_str or "BadRequestError" in error_str)
                    and ("context length" in error_str.lower() or "context_length" in error_str.lower()
                         or "maximum context" in error_str.lower() or "input_tokens" in error_str.lower())
                )
                if is_context_length_error:
                    if _truncated:
                        raise
                    _truncated = True
                    m_limit = _re.search(r'maximum context length is (\d+)', error_str)
                    m_input = _re.search(r'prompt contains at least (\d+) input tokens', error_str)
                    if m_limit and m_input:
                        model_limit = int(m_limit.group(1))
                        input_tokens = int(m_input.group(1))
                    else:
                        model_limit = self._max_model_len
                        input_tokens = len(prompt) // 4
                    target_input_tokens = max(256, model_limit - max_tokens)
                    scale = target_input_tokens / max(input_tokens, 1)
                    new_char_len = max(200, min(int(len(prompt) * scale), len(prompt) - 1))
                    head_len = int(new_char_len * 0.5)
                    old_len = len(prompt)
                    prompt = prompt[:head_len] + "\n...[truncated]...\n" + prompt[-(new_char_len - head_len):]
                    print(f"Context length exceeded, truncating prompt {old_len} -> ~{len(prompt)} chars, retrying...")
                    continue
                # Don't retry on other 400/client errors (permanent)
                if "400" in error_str or "BadRequestError" in error_str:
                    raise
                # Check if it's a rate limit error that should be retried
                attempt += 1
                if attempt >= max_retries:
                    raise
                if "rate" in error_str.lower() or "429" in error_str:
                    wait_time = min(rate_limit_base_wait * (2 ** (attempt - 1)), rate_limit_max_wait)
                    if rate_limit_jitter > 0:
                        wait_time += random.uniform(0, rate_limit_jitter * wait_time)
                    print(f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    wait_time = min(2 ** attempt, 30)
                    print(f"Error: {error_str}")
                    print(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

        raise RuntimeError(f"Failed after {max_retries} retries")
