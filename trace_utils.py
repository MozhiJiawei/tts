import atexit
import os
import threading
from collections import deque
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

TRACE_LANE_REQUEST = "request"
TRACE_LANE_BERT = "bert"
TRACE_LANE_LLM = "llm"
TRACE_LANE_FLOW_MATCHING = "flow_matching"

_TRACE_SLOT_COUNT = 7
_REQUEST_BASE_TID = 2400
_BERT_BASE_TID = 2100
_LLM_BASE_TID = 2200
_FLOW_MATCHING_TID = 2300

_REQUEST_TRACE_NAME = "put_request -> wait_audio"
_BERT_TRACE_NAME = "run_llm.preprocess -> submit_async"
_LLM_TRACE_NAME = "llm.process_tokens_and_poll"

_TRACE_ENABLED = str(os.getenv("VOICEBOX_TRACE", "")).strip().lower() in _TRUE_VALUES
_TRACE_PATH = os.getenv("VOICEBOX_TRACE_PATH", _DEFAULT_TRACE_PATH)

try:
    _TRACE_FLUSH_INTERVAL_S = float(os.getenv("VOICEBOX_TRACE_FLUSH_INTERVAL_S", _DEFAULT_FLUSH_INTERVAL_S))
except (TypeError, ValueError):
    _TRACE_FLUSH_INTERVAL_S = _DEFAULT_FLUSH_INTERVAL_S

_BOOTSTRAPPED = False
_LOCK = threading.Lock()
_TRACE_INTERFACE_LOGS: set[str] = set()
_STREAM_SLOT_BY_ID: dict[str, int] = {}
_SLOT_STREAM_IDS = [set() for _ in range(_TRACE_SLOT_COUNT)]
_FREE_SLOT_IDS = deque(range(_TRACE_SLOT_COUNT))
_FREE_SLOT_LOOKUP = set(range(_TRACE_SLOT_COUNT))


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
        tracer.configure_autosave(_TRACE_PATH, interval_s=_TRACE_FLUSH_INTERVAL_S)
        _configure_thread_metadata(tracer)
        atexit.register(_save_trace_at_exit)
        _BOOTSTRAPPED = True
        return True


def bind_stream_trace(stream_id: str) -> int:
    _print_trace_interface_call("bind_stream_trace")
    if not _TRACE_ENABLED:
        return -1

    bootstrap_tracing()
    with _LOCK:
        slot_id = _STREAM_SLOT_BY_ID.get(stream_id)
        if slot_id is None:
            slot_id = _acquire_slot_id(stream_id)
        return slot_id


def release_stream_trace(stream_id: str) -> None:
    _print_trace_interface_call("release_stream_trace")
    if not _TRACE_ENABLED:
        return

    with _LOCK:
        slot_id = _STREAM_SLOT_BY_ID.pop(stream_id, None)
        if slot_id is None:
            return
        stream_ids = _SLOT_STREAM_IDS[slot_id]
        stream_ids.discard(stream_id)
        if not stream_ids and slot_id not in _FREE_SLOT_LOOKUP:
            _FREE_SLOT_IDS.append(slot_id)
            _FREE_SLOT_LOOKUP.add(slot_id)


def begin_request_lifecycle(
    stream_id: str,
    name: str = _REQUEST_TRACE_NAME,
    args: Optional[dict[str, Any]] = None,
) -> None:
    _print_trace_interface_call("begin_request_lifecycle")
    if not _TRACE_ENABLED:
        return

    context = _get_stream_context(stream_id, create=True)
    get_tracer().begin(
        name,
        cat=TRACE_LANE_REQUEST,
        args=_build_stream_args(stream_id, args, context["slot"]),
        tid=context["request_tid"],
    )


def end_request_lifecycle(
    stream_id: str,
    name: str = _REQUEST_TRACE_NAME,
    args: Optional[dict[str, Any]] = None,
) -> None:
    _print_trace_interface_call("end_request_lifecycle")
    if not _TRACE_ENABLED:
        return

    context = _get_stream_context(stream_id, create=False)
    if context is None:
        return

    try:
        get_tracer().end(
            name,
            cat=TRACE_LANE_REQUEST,
            args=_build_stream_args(stream_id, args, context["slot"]),
            tid=context["request_tid"],
        )
    finally:
        release_stream_trace(stream_id)


def trace_request_instant(
    stream_id: str,
    name: str,
    args: Optional[dict[str, Any]] = None,
) -> None:
    _print_trace_interface_call("trace_request_instant")
    if not _TRACE_ENABLED:
        return

    context = _get_stream_context(stream_id, create=False)
    if context is None:
        return
    get_tracer().instant(
        name,
        cat=TRACE_LANE_REQUEST,
        args=_build_stream_args(stream_id, args, context["slot"]),
        tid=context["request_tid"],
    )


@contextmanager
def trace_stream_span(
    stream_id: str,
    lane: str,
    name: str,
    cat: str = "",
    args: Optional[dict[str, Any]] = None,
):
    _print_trace_interface_call("trace_stream_span")
    if not _TRACE_ENABLED:
        yield
        return

    context = _get_stream_context(stream_id, create=True)
    with get_tracer().complete_span(
        name,
        cat=cat or lane,
        args=_build_stream_args(stream_id, args, context["slot"]),
        tid=_get_stream_lane_tid(context, lane),
    ):
        yield


def trace_stream_instant(
    stream_id: str,
    lane: str,
    name: str,
    cat: str = "",
    args: Optional[dict[str, Any]] = None,
) -> None:
    _print_trace_interface_call("trace_stream_instant")
    if not _TRACE_ENABLED:
        return

    context = _get_stream_context(stream_id, create=False)
    if context is None:
        return
    get_tracer().instant(
        name,
        cat=cat or lane,
        args=_build_stream_args(stream_id, args, context["slot"]),
        tid=_get_stream_lane_tid(context, lane),
    )


@contextmanager
def trace_flow_matching_span(name: str, cat: str = "", args: Optional[dict[str, Any]] = None):
    _print_trace_interface_call("trace_flow_matching_span")
    if not _TRACE_ENABLED:
        yield
        return

    bootstrap_tracing()
    with get_tracer().complete_span(name, cat=cat or TRACE_LANE_FLOW_MATCHING, args=args, tid=_FLOW_MATCHING_TID):
        yield


def trace_flow_matching_instant(name: str, cat: str = "", args: Optional[dict[str, Any]] = None) -> None:
    _print_trace_interface_call("trace_flow_matching_instant")
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    get_tracer().instant(name, cat=cat or TRACE_LANE_FLOW_MATCHING, args=args, tid=_FLOW_MATCHING_TID)


@contextmanager
def trace_span(name: str, cat: str = "", args: Optional[dict[str, Any]] = None, lane: Optional[str] = None):
    _print_trace_interface_call("trace_span")
    if not _TRACE_ENABLED:
        yield
        return

    bootstrap_tracing()
    tid = _FLOW_MATCHING_TID if lane == TRACE_LANE_FLOW_MATCHING else None
    span_manager = get_tracer().complete_span if tid is not None else get_tracer().span
    with span_manager(name, cat=cat, args=args, tid=tid):
        yield


def trace_instant(name: str, cat: str = "", args: Optional[dict[str, Any]] = None, lane: Optional[str] = None) -> None:
    _print_trace_interface_call("trace_instant")
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    tid = _FLOW_MATCHING_TID if lane == TRACE_LANE_FLOW_MATCHING else None
    get_tracer().instant(name, cat=cat, args=args, tid=tid)


def trace_thread_name(name: str) -> None:
    _print_trace_interface_call("trace_thread_name")
    if not _TRACE_ENABLED:
        return

    bootstrap_tracing()
    get_tracer().set_thread_name(name)


@contextmanager
def trace_lane_span(lane: str, name: str, cat: str = "", args: Optional[dict[str, Any]] = None):
    if lane == TRACE_LANE_FLOW_MATCHING:
        with trace_flow_matching_span(name, cat=cat, args=args):
            yield
        return
    with trace_span(name, cat=cat, args=args, lane=lane):
        yield


def trace_lane_instant(lane: str, name: str, cat: str = "", args: Optional[dict[str, Any]] = None) -> None:
    if lane == TRACE_LANE_FLOW_MATCHING:
        trace_flow_matching_instant(name, cat=cat, args=args)
        return
    trace_instant(name, cat=cat, args=args, lane=lane)


def _configure_thread_metadata(tracer: Any) -> None:
    for slot_id in range(_TRACE_SLOT_COUNT):
        sort_base = slot_id * 3
        tracer.set_thread_name(f"RequestLifecycle[{slot_id}]", tid=_REQUEST_BASE_TID + slot_id)
        tracer.set_thread_name(f"RequestLifecycle[{slot_id}]/BERT", tid=_BERT_BASE_TID + slot_id)
        tracer.set_thread_name(f"RequestLifecycle[{slot_id}]/LLM", tid=_LLM_BASE_TID + slot_id)
        if hasattr(tracer, "set_thread_sort_index"):
            tracer.set_thread_sort_index(sort_base, tid=_REQUEST_BASE_TID + slot_id)
            tracer.set_thread_sort_index(sort_base + 1, tid=_BERT_BASE_TID + slot_id)
            tracer.set_thread_sort_index(sort_base + 2, tid=_LLM_BASE_TID + slot_id)

    tracer.set_thread_name("FlowMatching", tid=_FLOW_MATCHING_TID)
    if hasattr(tracer, "set_thread_sort_index"):
        tracer.set_thread_sort_index(_TRACE_SLOT_COUNT * 3, tid=_FLOW_MATCHING_TID)


def _acquire_slot_id(stream_id: str) -> int:
    if _FREE_SLOT_IDS:
        slot_id = _FREE_SLOT_IDS.popleft()
        _FREE_SLOT_LOOKUP.discard(slot_id)
    else:
        slot_id = abs(hash(stream_id)) % _TRACE_SLOT_COUNT
    _STREAM_SLOT_BY_ID[stream_id] = slot_id
    _SLOT_STREAM_IDS[slot_id].add(stream_id)
    return slot_id


def _get_stream_context(stream_id: str, create: bool) -> Optional[dict[str, int]]:
    if not _TRACE_ENABLED:
        return None

    bootstrap_tracing()
    with _LOCK:
        slot_id = _STREAM_SLOT_BY_ID.get(stream_id)
        if slot_id is None:
            if not create:
                return None
            slot_id = _acquire_slot_id(stream_id)
        return {
            "slot": slot_id,
            "request_tid": _REQUEST_BASE_TID + slot_id,
            "bert_tid": _BERT_BASE_TID + slot_id,
            "llm_tid": _LLM_BASE_TID + slot_id,
        }


def _get_stream_lane_tid(context: dict[str, int], lane: str) -> int:
    if lane == TRACE_LANE_REQUEST:
        return context["request_tid"]
    if lane == TRACE_LANE_BERT:
        return context["bert_tid"]
    if lane == TRACE_LANE_LLM:
        return context["llm_tid"]
    raise ValueError(f"unsupported stream trace lane: {lane}")


def _build_stream_args(stream_id: str, args: Optional[dict[str, Any]], slot_id: int) -> dict[str, Any]:
    return make_trace_args(args, stream_id=stream_id, slot=slot_id)


def _print_trace_interface_call(interface_name: str) -> None:
    with _LOCK:
        if interface_name in _TRACE_INTERFACE_LOGS:
            return
        _TRACE_INTERFACE_LOGS.add(interface_name)
    print(f"[trace_utils] {interface_name} called")


def _save_trace_at_exit() -> None:
    if not _TRACE_ENABLED or not _BOOTSTRAPPED:
        return

    try:
        get_tracer().save(_TRACE_PATH)
    except Exception:
        pass
