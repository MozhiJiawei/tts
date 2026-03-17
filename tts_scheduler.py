import concurrent
from queue import Queue

import numpy as np
from src.util.logger import logger
from src.inference.globals import text_normalizer, CFG, DEVICE, result_queue_map
from src.schedule.schedule_engine import ScheduleEngine

try:
    from src.schedule.trace_utils import bootstrap_tracing, make_trace_args, trace_instant, trace_span, trace_thread_name
except ImportError:
    from trace_utils import bootstrap_tracing, make_trace_args, trace_instant, trace_span, trace_thread_name

bootstrap_tracing("voicebox-tts-scheduler")


class TtsScheduler:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        bootstrap_tracing("voicebox-tts-scheduler")

        # ????????1????0-20??
        self.volume_factor = 1 if self.cfg.postprocess.volume_factor is None else self.cfg.postprocess.volume_factor
        if self.volume_factor > 20:
            self.volume_factor = 20
        elif self.volume_factor < 0:
            self.volume_factor = 1

        self.schedule_engine = ScheduleEngine(self.cfg, self.device)
        self.schedule_executor = concurrent.futures.ThreadPoolExecutor(max_workers=7)

    def __call__(
            self,
            session_id: str,
            stream_id: str,
            target_text: str,
            prompt_text: str,
            prompt_semantic_codes: np.ndarray,
            prompt_audio_24k: np.ndarray,
            target_lang: str = "zh",
            prompt_lang: str = "en"
    ):
        trace_thread_name("tts-request-thread")
        request_args = make_trace_args(session_id=session_id, stream_id=stream_id)
        with trace_span("tts.request.lifecycle", cat="tts", args=request_args):
            self.put_request(session_id, stream_id, target_text, prompt_text,
                             prompt_semantic_codes, prompt_audio_24k,
                             target_lang, prompt_lang)

            for audio in self.wait_audio(stream_id):
                yield audio

    def put_request(
            self,
            session_id: str,
            stream_id: str,
            target_text: str,
            prompt_text: str,
            prompt_audio: bytes,
            target_lang: str = "zh",
            prompt_lang: str = "en"
    ):
        trace_thread_name("tts-request-thread")
        trace_args = make_trace_args(session_id=session_id, stream_id=stream_id)
        with trace_span("tts.request.enqueue", cat="tts", args=trace_args):
            logger.info(f'tts scheduler put_request. sessionId: {session_id}, streamId: {stream_id}, targetText: {target_text}')

            with trace_span("tts.normalize_text", cat="tts", args=trace_args):
                # ???
                target_text = text_normalizer.process(target_text, target_lang)
                prompt_text = text_normalizer.process(prompt_text, prompt_lang)
            logger.info(f'normalize result, target_text: {target_text}, prompt_text: {prompt_text}')

            with trace_span("tts.submit_llm", cat="tts", args=trace_args):
                # LLM??
                self.schedule_executor.submit(self.schedule_engine.run_llm, stream_id, target_text, prompt_text, prompt_audio)

            with trace_span("tts.create_result_queue", cat="tts", args=trace_args):
                result_queue_map[stream_id] = Queue()

    def wait_audio(self, stream_id: str):
        trace_thread_name("tts-request-thread")
        trace_args = make_trace_args(stream_id=stream_id)
        while True:
            with trace_span("tts.wait_audio_block", cat="tts", args=trace_args):
                audio = result_queue_map[stream_id].get()
            if audio is None:
                trace_instant("tts.stream_complete", cat="tts", args=trace_args)
                yield None
                del result_queue_map[stream_id]
                break

            if self.volume_factor != 1:
                with trace_span("tts.postprocess_volume", cat="tts", args=make_trace_args(trace_args, volume_factor=self.volume_factor)):
                    audio = audio * self.volume_factor
                    audio = np.clip(audio, -1.0, 1.0)
            trace_instant("tts.audio_chunk_ready", cat="tts", args=make_trace_args(trace_args, chunk_len=len(audio)))
            yield audio

    def __pad_2d_list_left_numpy(self, lst, max_len, pad_value=0):
        """?2D????????"""

        # ??????
        padded_array = np.full((len(lst), max_len), pad_value, dtype=type(lst[0][0]))
        # ???????
        for i, sublist in enumerate(lst):
            padded_array[i, -len(sublist):] = sublist
        return padded_array


scheduler = TtsScheduler(CFG, DEVICE)
