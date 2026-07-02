import contextlib
import contextvars
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


_usage_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "usage_context",
    default={},
)


class UsageLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def current_usage_context() -> Dict[str, Any]:
    return dict(_usage_context.get({}))


@contextlib.contextmanager
def usage_context(**updates: Any) -> Iterator[None]:
    base = dict(_usage_context.get({}))
    base.update({k: v for k, v in updates.items() if v is not None})
    token = _usage_context.set(base)
    try:
        yield
    finally:
        _usage_context.reset(token)


def usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return dict(usage)
    out = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        if hasattr(usage, key):
            out[key] = getattr(usage, key)
    return out


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))
