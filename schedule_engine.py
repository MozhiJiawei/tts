import time
from threading import Thread

import librosa
import numpy as np
import torch

from src.schedule.pools import PriorityTokenPool
from src.inference.globals import tts_infer, infer_context_pool, result_queue_map
from src.util.logger import logger
from src.util.infer_engine_manager import get_llm_config
from src.components.llm.llm_manager import LlMManager
from src.util.audio_llm import AudioLLM
from transformers import AutoModelForCausalLM
from src.util.tensor_streamer import TensorStreamer

try:
    from src.schedule.trace_utils import make_trace_args, trace_instant, trace_span, trace_thread_name
except ImportError:
    from trace_utils import make_trace_args, trace_instant, trace_span, trace_thread_name


class ScheduleEngine:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.token_pool = PriorityTokenPool(cfg, device)
        self.fm_batch_queue = self.token_pool.subscribe()
        self.fm_min_token_cnt = self.cfg.scheduler.fm_min_token_cnt

        self.llm_engine, llm_device = get_llm_config()
        if self.llm_engine == 'mindie_llm_manager':
            # device???llm_manager_config.json??
            self.llm_manager = LlMManager(self.cfg.model.llm.llm_manager_path, self.token_pool)
        elif self.llm_engine == 'mindie_serivce':
            self.audio_llm = AudioLLM(self.cfg.model.llm.url)
            if not self.audio_llm.check_url_availability():
                raise Exception(f'audio llm url: {self.cfg.model.llm.url} is not available')
        elif self.llm_engine == 'pytorch':
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.pretrained_model_path,
                device_map=llm_device,
                torch_dtype=torch.float16,
                trust_remote_code=False,
            )
        else:
            raise Exception(f'unsupported llm engine: {self.llm_engine}')

        self.fm_run_batch_thread = Thread(target=self.__run_batch_for_flow_matching, name='fm-batch-worker')
        self.fm_run_batch_thread.start()

    def run_llm(
            self,
            stream_id,
            target_text,
            prompt_text,
            prompt_audio: bytes
    ):
        trace_thread_name('schedule-worker')
        trace_args = make_trace_args(stream_id=stream_id, backend=self.llm_engine)
        with trace_span('llm.request_lifecycle', cat='llm', args=trace_args):
            logger.info(f'run_llm stream_id: {stream_id}')

            with trace_span('llm.audio_tokenizer_prepare', cat='llm', args=trace_args):
                prompt_audio_16k = np.frombuffer(prompt_audio, dtype=np.int16).astype(np.float32) / 32768.0

            with trace_span(
                'npu.audio_tokenizer_infer',
                cat='npu',
                args=make_trace_args(trace_args, model_stage='audio_tokenizer', npu=True),
            ):
                prompt_semantic_codes = tts_infer.audio_tokenizer([prompt_audio_16k])

            with trace_span('llm.resample_prompt_audio', cat='llm', args=trace_args):
                prompt_audio_24k = librosa.resample(prompt_audio_16k, orig_sr=16000, target_sr=24000)
                batch_prompt_audio_24k = np.vstack([prompt_audio_24k])

            with trace_span(
                'npu.extract_mel_features_infer',
                cat='npu',
                args=make_trace_args(trace_args, model_stage='extract_mel_features', npu=True),
            ):
                short_prompt_semantic_codes, prompt_mel_feats = tts_infer.extract_mel_features(
                    batch_prompt_audio_24k,
                    prompt_semantic_codes,
                )
            infer_context_pool[stream_id] = {
                'prompt_semantic_codes': short_prompt_semantic_codes[0].detach().cpu().numpy(),
                'prompt_mel_feats': prompt_mel_feats[0].detach().cpu().numpy(),
                'last_token_batch': None,
                'last_mel': None,
            }

            with trace_span('llm.tokenize_text_prepare', cat='llm', args=trace_args):
                target_texts = [target_text]

            with trace_span(
                'npu.text_tokenize_infer',
                cat='npu',
                args=make_trace_args(trace_args, model_stage='text_tokenize', npu=True),
            ):
                llm_input_ids = tts_infer.tokenize_text(target_texts, prompt_semantic_codes)

            logger.info(f'??LLM?? stream_id: {stream_id}')
            with trace_span('llm.dispatch_backend', cat='llm', args=trace_args):
                if self.llm_engine == 'mindie_llm_manager':
                    with trace_span('llm.mindie_manager_submit', cat='llm', args=trace_args):
                        self.llm_manager.async_forward(stream_id, llm_input_ids[0])
                elif self.llm_engine == 'mindie_serivce':
                    llm_inference_thread = Thread(
                        target=self.__llm_inference_mindie_service,
                        args=(stream_id, llm_input_ids[0]),
                        name=f'llm-worker-mindie_serivce-{stream_id}',
                    )
                    llm_inference_thread.start()
                    trace_instant('llm.worker_thread_started', cat='llm', args=trace_args)
                elif self.llm_engine == 'pytorch':
                    llm_inference_thread = Thread(
                        target=self.__llm_inference_transformers,
                        args=(stream_id, llm_input_ids[0]),
                        name=f'llm-worker-pytorch-{stream_id}',
                    )
                    llm_inference_thread.start()
                    trace_instant('llm.worker_thread_started', cat='llm', args=trace_args)
                else:
                    raise Exception(f'unsupported llm engine: {self.llm_engine}')

    def __llm_inference_mindie_service(self, stream_id, llm_input_ids, top_k=5, top_p=0.5, temp=0.3):
        trace_thread_name(f'llm-worker-mindie_serivce-{stream_id}')
        trace_args = make_trace_args(stream_id=stream_id, backend='mindie_serivce')
        is_first_token = True
        with trace_span(
            'npu.llm_stream_query',
            cat='npu',
            args=make_trace_args(trace_args, model_stage='llm_stream_query', npu=True),
        ):
            for token in self.audio_llm.stream_query(llm_input_ids, top_k, top_p, temp):
                is_eos = token == self.cfg.model.llm.eos
                token_data = {
                    'stream_id': stream_id,
                    'token': token,
                    'is_eos': is_eos,
                }
                if is_first_token:
                    trace_instant('llm.first_token', cat='llm', args=trace_args)
                    is_first_token = False
                trace_instant('llm.token_received', cat='llm', args=make_trace_args(trace_args, is_eos=is_eos))
                if is_eos:
                    trace_instant('llm.eos_received', cat='llm', args=trace_args)
                self.token_pool.put(token_data)

    def __llm_inference_transformers(self, stream_id, llm_input_ids, top_k=5, top_p=0.5, temp=0.3):
        trace_thread_name(f'llm-worker-pytorch-{stream_id}')
        trace_args = make_trace_args(stream_id=stream_id, backend='pytorch')
        with trace_span('llm.prepare_generation_inputs', cat='llm', args=trace_args):
            streamer = TensorStreamer()
            generation_kwargs = {
                'input_ids': torch.tensor([llm_input_ids]).to(self.device),
                'min_new_tokens': 12,
                'max_new_tokens': 400,
                'streamer': streamer,
                'do_sample': True,
                'top_k': top_k,
                'top_p': top_p,
                'temperature': temp,
                'eos_token_id': self.cfg.preprocess.end_token_id,
                'pad_token_id': self.cfg.preprocess.pad_token_id,
            }
        thread = Thread(
            target=self.__run_transformers_generate,
            args=(stream_id, generation_kwargs),
            name=f'npu-llm-generate-pytorch-{stream_id}',
        )
        thread.start()
        trace_instant('llm.generate_thread_started', cat='llm', args=trace_args)

        token_record = []
        is_first_payload_token = True
        for token_tensor in streamer:
            token = token_tensor.cpu().numpy()[0]
            if is_first_payload_token:
                is_first_payload_token = False
                continue
            is_eos = token == self.cfg.model.llm.eos
            token_record.append(token)
            token_data = {
                'stream_id': stream_id,
                'token': token,
                'is_eos': is_eos,
            }
            if len(token_record) == 1:
                trace_instant('llm.first_token', cat='llm', args=trace_args)
            trace_instant('llm.token_received', cat='llm', args=make_trace_args(trace_args, is_eos=is_eos))
            if is_eos:
                trace_instant('llm.eos_received', cat='llm', args=trace_args)
            self.token_pool.put(token_data)
        print('transformers tokens', token_record)
        thread.join()

    def __run_transformers_generate(self, stream_id, generation_kwargs):
        trace_thread_name(f'npu-llm-generate-pytorch-{stream_id}')
        with trace_span(
            'npu.llm_generate',
            cat='npu',
            args=make_trace_args(stream_id=stream_id, backend='pytorch', model_stage='llm_generate', npu=True),
        ):
            self.llm.generate(**generation_kwargs)

    def __run_batch_for_flow_matching(self):
        trace_thread_name('fm-batch-worker')
        while True:
            with trace_span('fm.wait_batch_queue', cat='fm'):
                fm_batch = self.fm_batch_queue.get()

            stream_ids = [fm['stream_id'] for fm in fm_batch]
            batch_args = make_trace_args(stream_ids=stream_ids, batch_size=len(stream_ids))
            with trace_span('fm.batch_cycle', cat='fm', args=batch_args):
                start_time = time.time()
                logger.info(f'FM?? stream_ids: {stream_ids}')

                with trace_span('fm.prepare_batch_inputs', cat='fm', args=batch_args):
                    batch_combine_batch_code = np.zeros((len(stream_ids), 25 + 2 + self.fm_min_token_cnt), dtype=np.int32)
                    batch_mel_cond = np.zeros((len(stream_ids), 108, 128), dtype=np.float32)

                    for idx, stream_id in enumerate(stream_ids):
                        infer_context = infer_context_pool[stream_id]
                        prompt_semantic_codes = infer_context['prompt_semantic_codes']
                        current_token_batch = fm_batch[idx]['tokens']

                        # ?????????token
                        if infer_context['last_token_batch'] is None:
                            combine_batch_code = np.concatenate([prompt_semantic_codes, current_token_batch], axis=0)
                        else:
                            combine_batch_code = np.concatenate(
                                [prompt_semantic_codes, infer_context['last_token_batch'], current_token_batch],
                                axis=0,
                            )

                        batch_combine_batch_code[idx, -len(combine_batch_code):] = combine_batch_code

                        current_mel = infer_context['prompt_mel_feats']

                        # ??mel??
                        if infer_context['last_mel'] is None:
                            mel_cond = current_mel
                        else:
                            mel_cond = np.concatenate([current_mel, infer_context['last_mel']], axis=0)

                        batch_mel_cond[idx, -len(mel_cond):] = mel_cond

                logger.info(f'FM?batch ??: {time.time() - start_time} seconds')

                with trace_span(
                    'npu.flow_matching_infer',
                    cat='npu',
                    args=make_trace_args(batch_args, model_stage='flow_matching', npu=True),
                ):
                    audio_list, batch_mel = tts_infer.code2audio(batch_combine_batch_code, batch_mel_cond)

                with trace_span('fm.emit_audio_results', cat='fm', args=batch_args):
                    for idx, fm in enumerate(fm_batch):
                        stream_id = fm['stream_id']
                        actual_token_cnt = fm['actual_token_cnt']
                        batch_token_cnt = fm['batch_token_cnt']
                        cur_audio = audio_list[idx][0]
                        slt_idx = int(actual_token_cnt / batch_token_cnt * len(cur_audio))
                        logger.info(f'???? stream_id: {stream_id}, len: {len(cur_audio[:slt_idx])}')
                        result_queue_map[stream_id].put(cur_audio[:slt_idx])
                        trace_instant(
                            'fm.audio_emitted',
                            cat='fm',
                            args=make_trace_args(stream_id=stream_id, chunk_len=len(cur_audio[:slt_idx]), has_eos=fm['has_eos']),
                        )
                        if fm['has_eos']:
                            # ??????token
                            logger.info(f'stream_id: {stream_id} ????')
                            trace_instant('fm.stream_finalized', cat='fm', args=make_trace_args(stream_id=stream_id))
                            result_queue_map[stream_id].put(None)
                            del infer_context_pool[stream_id]
                            self.token_pool.remove(stream_id)
                            continue

                        infer_context_pool[stream_id]['last_token_batch'] = fm['tokens'][-2:]
                        infer_context_pool[stream_id]['last_mel'] = batch_mel[idx, -2 * 4:, :]
                        trace_instant('fm.context_updated', cat='fm', args=make_trace_args(stream_id=stream_id))
                self.token_pool.notify_batch()
