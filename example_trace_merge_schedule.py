import os

os.environ.setdefault("VOICEBOX_TRACE", "1")
os.environ.setdefault("VOICEBOX_TRACE_PATH", "trace_merge_schedule_demo.json")
os.environ.setdefault("VOICEBOX_TRACE_FLUSH_INTERVAL_S", "60")

from chrome_tracing import get_tracer, reset_tracer
from trace_utils import make_trace_args

TRACE_PATH = os.environ["VOICEBOX_TRACE_PATH"]
SLOT_COUNT = 7
REQUEST_TIDS = [2400 + slot for slot in range(SLOT_COUNT)]
BERT_TIDS = [2100 + slot for slot in range(SLOT_COUNT)]
LLM_TIDS = [2200 + slot for slot in range(SLOT_COUNT)]
FLOW_TID = 2300


def ms_to_us(value_ms: int) -> int:
    return value_ms * 1000


def emit_complete(tracer, tid: int, name: str, start_ms: int, dur_ms: int, cat: str, **args) -> None:
    tracer.complete(
        name,
        dur_us=ms_to_us(dur_ms),
        cat=cat,
        args=args or None,
        tid=tid,
        ts_us=ms_to_us(start_ms),
    )


def emit_instant(tracer, tid: int, name: str, ts_ms: int, cat: str, **args) -> None:
    tracer.instant(name, cat=cat, args=args or None, tid=tid, ts_us=ms_to_us(ts_ms))


def build_request_specs() -> list[dict]:
    specs: list[dict] = []
    fm_cursor_ms = 430

    for slot in range(SLOT_COUNT):
        stream_id = f"stream-{slot + 1:03d}"
        request_start_ms = slot * 55
        bert_start_ms = request_start_ms + 24 + (slot % 2) * 8
        bert_dur_ms = 70 + (slot % 3) * 10
        llm_start_ms = bert_start_ms + bert_dur_ms + 28
        llm_dur_ms = 180 + (slot % 4) * 18

        fm_wait_start_ms = fm_cursor_ms
        fm_wait_dur_ms = 45 + (slot % 2) * 10
        fm_call_start_ms = fm_wait_start_ms + fm_wait_dur_ms + 8
        fm_call_dur_ms = 48 + (slot % 3) * 8
        fm_cursor_ms = fm_call_start_ms + fm_call_dur_ms + 20

        first_chunk_ts_ms = fm_call_start_ms + fm_call_dur_ms + 18
        request_end_ts_ms = first_chunk_ts_ms + 85 + (slot % 2) * 18
        request_dur_ms = request_end_ts_ms - request_start_ms

        specs.append(
            {
                "slot": slot,
                "stream_id": stream_id,
                "request_tid": REQUEST_TIDS[slot],
                "bert_tid": BERT_TIDS[slot],
                "llm_tid": LLM_TIDS[slot],
                "request_start_ms": request_start_ms,
                "request_dur_ms": request_dur_ms,
                "bert_start_ms": bert_start_ms,
                "bert_dur_ms": bert_dur_ms,
                "llm_start_ms": llm_start_ms,
                "llm_dur_ms": llm_dur_ms,
                "fm_wait_start_ms": fm_wait_start_ms,
                "fm_wait_dur_ms": fm_wait_dur_ms,
                "fm_call_start_ms": fm_call_start_ms,
                "fm_call_dur_ms": fm_call_dur_ms,
                "first_chunk_ts_ms": first_chunk_ts_ms,
                "request_end_ts_ms": request_end_ts_ms,
            }
        )

    return specs


def render_request_group(tracer, spec: dict) -> None:
    base_args = {"stream_id": spec["stream_id"], "slot": spec["slot"]}

    emit_complete(
        tracer,
        spec["request_tid"],
        "put_request -> wait_audio",
        spec["request_start_ms"],
        spec["request_dur_ms"],
        "request",
        **base_args,
    )
    emit_instant(tracer, spec["request_tid"], "put_request", spec["request_start_ms"], "request", **base_args)
    emit_instant(tracer, spec["request_tid"], "wait_audio.first_chunk", spec["first_chunk_ts_ms"], "request", **base_args)
    emit_instant(tracer, spec["request_tid"], "wait_audio.end", spec["request_end_ts_ms"], "request", **base_args)

    emit_complete(
        tracer,
        spec["bert_tid"],
        "run_llm.preprocess -> submit_async",
        spec["bert_start_ms"],
        spec["bert_dur_ms"],
        "bert",
        **make_trace_args(base_args, phase="bert_preprocess"),
    )

    emit_complete(
        tracer,
        spec["llm_tid"],
        "llm.process_tokens_and_poll",
        spec["llm_start_ms"],
        spec["llm_dur_ms"],
        "llm",
        **make_trace_args(base_args, phase="llm_process"),
    )
    emit_instant(
        tracer,
        spec["llm_tid"],
        "llm.token_batch_ready",
        spec["llm_start_ms"] + spec["llm_dur_ms"],
        "llm",
        **base_args,
    )


def render_flow_matching(tracer, specs: list[dict]) -> None:
    for spec in specs:
        base_args = {"stream_id": spec["stream_id"], "slot": spec["slot"]}
        emit_complete(
            tracer,
            FLOW_TID,
            "fm.wait_batch",
            spec["fm_wait_start_ms"],
            spec["fm_wait_dur_ms"],
            "flow_matching",
            **make_trace_args(base_args, phase="wait_batch"),
        )
        emit_complete(
            tracer,
            FLOW_TID,
            "fm.call_flowMatching",
            spec["fm_call_start_ms"],
            spec["fm_call_dur_ms"],
            "flow_matching",
            **make_trace_args(base_args, phase="call_flowMatching"),
        )


def configure_threads(tracer) -> None:
    for slot in range(SLOT_COUNT):
        sort_base = slot * 3
        tracer.set_thread_name(f"RequestLifecycle[{slot}]", tid=REQUEST_TIDS[slot])
        tracer.set_thread_sort_index(sort_base, tid=REQUEST_TIDS[slot])

        tracer.set_thread_name(f"BERT[{slot}]", tid=BERT_TIDS[slot])
        tracer.set_thread_sort_index(sort_base + 1, tid=BERT_TIDS[slot])

        tracer.set_thread_name(f"LLM[{slot}]", tid=LLM_TIDS[slot])
        tracer.set_thread_sort_index(sort_base + 2, tid=LLM_TIDS[slot])

    tracer.set_thread_name("FlowMatching", tid=FLOW_TID)
    tracer.set_thread_sort_index(SLOT_COUNT * 3, tid=FLOW_TID)


def main() -> None:
    reset_tracer()
    tracer = get_tracer()
    tracer.set_process_name("voicebox-schedule-7-slot-demo")
    configure_threads(tracer)

    request_specs = build_request_specs()
    for request_spec in request_specs:
        render_request_group(tracer, request_spec)
    render_flow_matching(tracer, request_specs)

    tracer.save(TRACE_PATH)
    print(f"trace saved to: {TRACE_PATH}")


if __name__ == "__main__":
    main()
