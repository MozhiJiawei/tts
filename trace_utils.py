import atexit
import os
import threading
from contextlib import contextmanager
from typing import Any, Optional

try:
    from src.schedule.chrome_tracing import get_tracer, reset_tracer
except ImportError:
    from chrome_tracing import get_tracer, reset_tracer

_DEFAULT_PROCESS_NAME = "voicebox-tts-scheduler"
_DEFAULT_TRACE_PATH = "trace.json"
_TRUE_VALUES = {"1", "true", "yes", "on"}

_TRACE_ENABLED = str(os.getenv("VOICEBOX_TRACE", "")).strip().lower() in _TRUE_VALUES
_TRACE_PATH = os.getenv("VOICEBOX_TRACE_PATH", _DEFAULT_TRACE_PATH)
_BOOTSTRAPPED = False
_LOCK = threading.Lock()


def is_trace_enabled() -> bool:
    return _TRACE_ENABLED


def make_trace_args(*sources: Optional[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        if not source:
            continue
        for key, value in source.items():
            if value is not None:
                merged[key] = value
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value
    return merged


def bootstrap_tracing(process_name: str = _DEFAULT_PROCESS_NAME) -> bool:
    global _BOOTSTRAPPED
    if not _TRACE_ENABLED:
        return False

    with _LOCK:
        if _BOOTSTRAPPED:
            return True
        reset_tracer()
        tracer = get_tracer()
        tracer.set_process_name(process_name)
        atexit.register(_save_trace_at_exit)
        _BOOTSTRAPPED = True
        return True


@contextmanager
def trace_span(name: str, cat: str = "", args: Optional[dict[str, Any]] = None):
    if not _TRACE_ENABLED:
        yield
        return

    bootstrap_tracing()
    with get_tracer().span(name, cat=cat, args=args):
        yield


def trace_instant(name: str, cat: str = "", args: Optional[dict[str, Any]] = None) -> None:
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    get_tracer().instant(name, cat=cat, args=args)


def trace_thread_name(name: str) -> None:
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    get_tracer().set_thread_name(name)


def _save_trace_at_exit() -> None:
    if not _TRACE_ENABLED or not _BOOTSTRAPPED:
        return

    try:
        get_tracer().save(_TRACE_PATH)
    except Exception:
        pass
