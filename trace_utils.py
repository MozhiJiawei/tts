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
_DEFAULT_FLUSH_INTERVAL_S = 1.0
_TRUE_VALUES = {"1", "true", "yes", "on"}

TRACE_LANE_BERT = "bert"
TRACE_LANE_LLM = "llm"
TRACE_LANE_FLOW_MATCHING = "flow_matching"

_LOGICAL_LANE_TIDS = {
    TRACE_LANE_BERT: 2001,
    TRACE_LANE_LLM: 2002,
    TRACE_LANE_FLOW_MATCHING: 2003,
}

_LOGICAL_LANE_NAMES = {
    TRACE_LANE_BERT: "BERT",
    TRACE_LANE_LLM: "LLM",
    TRACE_LANE_FLOW_MATCHING: "FlowMatching",
}

_TRACE_ENABLED = str(os.getenv("VOICEBOX_TRACE", "")).strip().lower() in _TRUE_VALUES
_TRACE_PATH = os.getenv("VOICEBOX_TRACE_PATH", _DEFAULT_TRACE_PATH)

try:
    _TRACE_FLUSH_INTERVAL_S = float(os.getenv("VOICEBOX_TRACE_FLUSH_INTERVAL_S", _DEFAULT_FLUSH_INTERVAL_S))
except (TypeError, ValueError):
    _TRACE_FLUSH_INTERVAL_S = _DEFAULT_FLUSH_INTERVAL_S
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


def get_trace_lane_tid(lane: str) -> int:
    try:
        return _LOGICAL_LANE_TIDS[lane]
    except KeyError as exc:
        raise ValueError(f"unsupported trace lane: {lane}") from exc


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
        tracer.configure_autosave(_TRACE_PATH, interval_s=_TRACE_FLUSH_INTERVAL_S)
        for lane, tid in _LOGICAL_LANE_TIDS.items():
            tracer.set_thread_name(_LOGICAL_LANE_NAMES[lane], tid=tid)
        atexit.register(_save_trace_at_exit)
        _BOOTSTRAPPED = True
        return True


@contextmanager
def trace_span(name: str, cat: str = "", args: Optional[dict[str, Any]] = None, lane: Optional[str] = None):
    if not _TRACE_ENABLED:
        yield
        return

    bootstrap_tracing()
    tracer = get_tracer()
    tid = get_trace_lane_tid(lane) if lane is not None else None
    span_manager = tracer.complete_span if tid is not None else tracer.span
    with span_manager(name, cat=cat, args=args, tid=tid):
        yield


def trace_instant(name: str, cat: str = "", args: Optional[dict[str, Any]] = None, lane: Optional[str] = None) -> None:
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    tid = get_trace_lane_tid(lane) if lane is not None else None
    get_tracer().instant(name, cat=cat, args=args, tid=tid)


def trace_thread_name(name: str) -> None:
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    get_tracer().set_thread_name(name)


@contextmanager
def trace_lane_span(lane: str, name: str, cat: str = "", args: Optional[dict[str, Any]] = None):
    with trace_span(name, cat=cat, args=args, lane=lane):
        yield


def trace_lane_instant(lane: str, name: str, cat: str = "", args: Optional[dict[str, Any]] = None) -> None:
    trace_instant(name, cat=cat, args=args, lane=lane)


def _save_trace_at_exit() -> None:
    if not _TRACE_ENABLED or not _BOOTSTRAPPED:
        return

    try:
        get_tracer().save(_TRACE_PATH)
    except Exception:
        pass
