from queue import Queue
from src.util.logger import logger


class PriorityTokenPool:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.llm_eos = self.cfg.model.llm.eos
        self.max_batch_size = self.cfg.scheduler.fm_max_batch_size
        self.min_batch_size = self.cfg.scheduler.fm_min_batch_size
        self.min_token_cnt = self.cfg.scheduler.fm_min_token_cnt
        self.audio_token_shift = self.cfg.preprocess.audio_token_shift

        self.pool = {}
        self.batch_queue = Queue()
        self.eos_stream_ids = set()

    def put(self, token_data):
        stream_id = token_data['stream_id']
        token = token_data['token'] - self.audio_token_shift
        if stream_id not in self.pool:
            logger.info(f'first token arrived: {stream_id}')
            self.pool[stream_id] = Queue()
        self.pool[stream_id].put({'token': token, 'is_eos': token_data['is_eos']})
        if token_data['is_eos']:
            logger.info(f'LLM output finished: {stream_id}')
            self.eos_stream_ids.add(stream_id)

        # Wake the FM worker when the first ready data reaches the pool.
        if self.batch_queue.empty():
            self.notify_batch()

    def notify_batch(self):
        token_stat = {
            'enough': set(),
            'close': set(),
            'little': set(),
        }
        for stream_id in self.pool.keys():
            token_cnt = self.pool[stream_id].qsize()

            # Treat EOS streams as batchable so the tail chunk does not wait forever.
            if token_cnt >= self.min_token_cnt or stream_id in self.eos_stream_ids:
                token_stat['enough'].add(stream_id)
            elif token_cnt > int(self.min_token_cnt * 0.5):
                token_stat['close'].add(stream_id)
            else:
                token_stat['little'].add(stream_id)

        if len(token_stat['enough']) < self.min_batch_size:
            return

        for stream_id in self.pool.keys():
            logger.info(f'token pool state stream_id: {stream_id}, queue_size: {self.pool[stream_id].qsize()}')

        # If the batch is not full yet, prefer waiting for near-ready streams.
        if len(token_stat['enough']) < self.max_batch_size and len(token_stat['close']) > 0:
            logger.info(f'wait for near-threshold token streams: {token_stat["close"]}')
            return

        batch = []
        for stream_id in token_stat['enough']:
            tokens = []
            has_eos = False
            for _ in range(self.min_token_cnt):
                if self.pool[stream_id].empty():
                    break
                token_data = self.pool[stream_id].get()
                tokens.append(token_data['token'])
                if token_data['is_eos']:
                    has_eos = True
            if has_eos:
                self.eos_stream_ids.remove(stream_id)

            actual_token_cnt = len(tokens)
            if actual_token_cnt < self.min_token_cnt:
                tokens.extend([self.llm_eos] * (self.min_token_cnt - len(tokens)))
            batch.append({
                'stream_id': stream_id,
                'actual_token_cnt': actual_token_cnt,
                'batch_token_cnt': self.min_token_cnt,
                'tokens': tokens,
                'has_eos': has_eos,
            })
            if len(batch) >= self.max_batch_size:
                break
        self.batch_queue.put(batch)

        stream_ids = [req['stream_id'] for req in batch]
        logger.info(f'batch_queue put new batch: {stream_ids}, size: {self.batch_queue.qsize()}')

    def subscribe(self):
        return self.batch_queue

    def remove(self, stream_id: str):
        del self.pool[stream_id]
