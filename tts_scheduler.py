import concurrent
from queue import Queue

import numpy as np
from src.util.logger import logger
from src.inference.globals import text_normalizer, CFG, DEVICE, result_queue_map
from src.schedule.schedule_engine import ScheduleEngine
from src.schedule.trace_utils import begin_request_lifecycle, end_request_lifecycle, make_trace_args


class TtsScheduler:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # Default volume gain is 1 and is clamped to the range [0, 20].
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
        logger.info(f'tts scheduler put_request. sessionId: {session_id}, streamId: {stream_id}, targetText: {target_text}')
        trace_args = make_trace_args(
            session_id=session_id,
            target_lang=target_lang,
            prompt_lang=prompt_lang,
        )
        begin_request_lifecycle(stream_id, args=trace_args)

        try:
            # Normalize text before submitting the LLM job.
            target_text = text_normalizer.process(target_text, target_lang)
            prompt_text = text_normalizer.process(prompt_text, prompt_lang)
            logger.info(f'normalize result, target_text: {target_text}, prompt_text: {prompt_text}')

            # Submit the LLM job and create the audio result queue.
            self.schedule_executor.submit(self.schedule_engine.run_llm, stream_id, target_text, prompt_text, prompt_audio)
            result_queue_map[stream_id] = Queue()
        except Exception as exc:
            end_request_lifecycle(
                stream_id,
                args=make_trace_args(trace_args, status="submit_failed", error=type(exc).__name__),
            )
            raise

    def wait_audio(self, stream_id: str):
        request_trace_closed = False

        try:
            while True:
                audio = result_queue_map[stream_id].get()
                if audio is None:
                    end_request_lifecycle(stream_id, args={"status": "completed"})
                    request_trace_closed = True
                    del result_queue_map[stream_id]
                    yield None
                    break

                if self.volume_factor != 1:
                    audio = audio * self.volume_factor
                    audio = np.clip(audio, -1.0, 1.0)
                yield audio
        finally:
            if not request_trace_closed:
                end_request_lifecycle(stream_id, args={"status": "closed_early"})

    def __pad_2d_list_left_numpy(self, lst, max_len, pad_value=0):
        """Left-pad a 2D list and convert it to a NumPy array."""

        # Build the target array and align each row on the right.
        padded_array = np.full((len(lst), max_len), pad_value, dtype=type(lst[0][0]))
        for i, sublist in enumerate(lst):
            padded_array[i, -len(sublist):] = sublist
        return padded_array


scheduler = TtsScheduler(CFG, DEVICE)
