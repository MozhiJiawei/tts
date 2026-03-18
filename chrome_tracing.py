"""
Chrome Tracing 打点工具

用于在 Python 多线程/异步代码中打点，生成可在 Chrome 中查看的时间线。
使用方式：在 Chrome 地址栏打开 chrome://tracing ，点击 Load 加载生成的 .json 文件。

事件类型说明：
- B/E (Begin/End): 成对标记一段区间的开始和结束，用于时长统计
- X (Complete): 单条记录表示完整区间（含 dur），无需成对
- i (Instant): 瞬时事件，无时长
- M (Metadata): 进程/线程名称等元数据
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional


# 全局 tracer 实例与锁
_tracer: Optional["ChromeTracer"] = None
_lock = threading.Lock()


def get_tracer() -> "ChromeTracer":
    """获取或创建全局 tracer（线程安全）。"""
    global _tracer
    with _lock:
        if _tracer is None:
            _tracer = ChromeTracer()
        return _tracer


def reset_tracer() -> None:
    """重置全局 tracer（常用于新一次录制）。"""
    global _tracer
    old_tracer: Optional["ChromeTracer"] = None
    with _lock:
        old_tracer = _tracer
        _tracer = None
    if old_tracer is not None:
        old_tracer.close()


class ChromeTracer:
    """
    Chrome Trace Event 格式的录制器。

    时间戳使用相对起点（微秒），便于在 chrome://tracing 中查看。
    所有写操作线程安全。
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._start_ns = time.perf_counter_ns()
        self._lock = threading.Lock()
        self._pid = 1  # 可选：用 os.getpid()
        self._dirty = False
        self._event_version = 0
        self._autosave_path: Optional[str] = None
        self._autosave_interval_s = 1.0
        self._autosave_thread: Optional[threading.Thread] = None
        self._stop_autosave = threading.Event()

    def _ts_us(self) -> int:
        """当前相对起点的时间（微秒）。"""
        return (time.perf_counter_ns() - self._start_ns) // 1000

    def _tid(self) -> int:
        """当前线程 ID（用于区分线程）。"""
        return threading.get_ident()

    def _emit(
        self,
        ph: str,
        name: str,
        cat: str = "",
        args: Optional[dict] = None,
        dur_us: Optional[int] = None,
        tid: Optional[int] = None,
        ts_us: Optional[int] = None,
    ) -> None:
        """发送一条 trace 事件。"""
        ts = self._ts_us() if ts_us is None else ts_us
        ev: dict[str, Any] = {
            "name": name,
            "cat": cat,
            "ph": ph,
            "ts": ts,
            "pid": self._pid,
        }
        ev["tid"] = tid if tid is not None else self._tid()
        if args:
            ev["args"] = args
        if dur_us is not None and ph == "X":
            ev["dur"] = dur_us
        with self._lock:
            self._events.append(ev)
            self._dirty = True
            self._event_version += 1

    def begin(self, name: str, cat: str = "", args: Optional[dict] = None, tid: Optional[int] = None) -> None:
        """区间开始 (Phase B)。"""
        self._emit("B", name, cat=cat, args=args, tid=tid)

    def end(self, name: str, cat: str = "", args: Optional[dict] = None, tid: Optional[int] = None) -> None:
        """区间结束 (Phase E)。"""
        self._emit("E", name, cat=cat, args=args, tid=tid)

    def complete(
        self,
        name: str,
        dur_us: int,
        cat: str = "",
        args: Optional[dict] = None,
        tid: Optional[int] = None,
        ts_us: Optional[int] = None,
    ) -> None:
        """完整区间 (Phase X)，一条记录包含时长。"""
        self._emit("X", name, cat=cat, args=args, dur_us=dur_us, tid=tid, ts_us=ts_us)

    def instant(self, name: str, cat: str = "", args: Optional[dict] = None, tid: Optional[int] = None) -> None:
        """瞬时事件 (Phase i)。"""
        self._emit("i", name, cat=cat, args=args, tid=tid)

    def metadata(self, name: str, args: dict, tid: Optional[int] = None) -> None:
        """元数据 (Phase M)，如 process_name、thread_name。"""
        self._emit("M", name, args=args, tid=tid)

    def set_process_name(self, name: str) -> None:
        """设置进程显示名称。"""
        self.metadata("process_name", {"name": name})

    def set_thread_name(self, name: str, tid: Optional[int] = None) -> None:
        """设置线程显示名称（在 chrome://tracing 中会显示）。"""
        # Chrome 通过事件的 tid 关联线程，args 只需 name
        self.metadata("thread_name", {"name": name}, tid=tid)

    @contextmanager
    def span(self, name: str, cat: str = "", args: Optional[dict] = None, tid: Optional[int] = None):
        """上下文管理器：自动记录 begin/end 区间。"""
        self.begin(name, cat=cat, args=args, tid=tid)
        try:
            yield
        finally:
            self.end(name, cat=cat, args=args, tid=tid)

    @contextmanager
    def complete_span(self, name: str, cat: str = "", args: Optional[dict] = None, tid: Optional[int] = None):
        """上下文管理器：退出时记录一条完整区间 X 事件。"""
        start_us = self._ts_us()
        try:
            yield
        finally:
            self.complete(name, self._ts_us() - start_us, cat=cat, args=args, tid=tid, ts_us=start_us)

    def record_duration(self, name: str, dur_us: int, cat: str = "", args: Optional[dict] = None) -> None:
        """已知时长时，直接记录一条 X 事件。"""
        self.complete(name, dur_us, cat=cat, args=args)

    def to_json(self) -> str:
        """导出为 Chrome 可加载的 JSON 字符串。"""
        with self._lock:
            return json.dumps(self._events, ensure_ascii=False)

    def save(self, path: str) -> None:
        """保存到文件，可直接在 chrome://tracing 中 Load。"""
        with self._lock:
            payload = json.dumps(self._events, ensure_ascii=False)
            version = self._event_version
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
        with self._lock:
            if self._event_version == version:
                self._dirty = False

    def clear(self) -> None:
        """清空已记录事件（不重置时间起点）。"""
        with self._lock:
            self._events.clear()
            self._dirty = False
            self._event_version += 1

    def configure_autosave(self, path: str, interval_s: float = 1.0) -> None:
        """配置后台周期保存，避免只在进程退出时落盘。"""
        normalized_interval = max(float(interval_s), 0.1)
        with self._lock:
            self._autosave_path = path
            self._autosave_interval_s = normalized_interval
            if self._autosave_thread is not None and self._autosave_thread.is_alive():
                return
            self._stop_autosave.clear()
            self._autosave_thread = threading.Thread(
                target=self._autosave_loop,
                name="trace-autosave",
                daemon=True,
            )
            self._autosave_thread.start()

    def flush(self) -> None:
        """立刻将当前 trace 保存到配置路径。"""
        with self._lock:
            path = self._autosave_path
            dirty = self._dirty
        if path and dirty:
            self.save(path)

    def close(self) -> None:
        """停止后台 autosave，并尝试执行最后一次刷新。"""
        thread: Optional[threading.Thread]
        with self._lock:
            thread = self._autosave_thread
        self._stop_autosave.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        try:
            self.flush()
        except Exception:
            pass

    def _autosave_loop(self) -> None:
        while not self._stop_autosave.wait(self._autosave_interval_s):
            try:
                self.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 便捷函数（使用全局 tracer）
# ---------------------------------------------------------------------------

def begin(name: str, cat: str = "", args: Optional[dict] = None) -> None:
    get_tracer().begin(name, cat=cat, args=args)


def end(name: str, cat: str = "", args: Optional[dict] = None) -> None:
    get_tracer().end(name, cat=cat, args=args)


def instant(name: str, cat: str = "", args: Optional[dict] = None) -> None:
    get_tracer().instant(name, cat=cat, args=args)


@contextmanager
def span(name: str, cat: str = "", args: Optional[dict] = None):
    """全局 span：with span('任务名'): ..."""
    t = get_tracer()
    t.begin(name, cat=cat, args=args)
    try:
        yield
    finally:
        t.end(name, cat=cat, args=args)


def set_thread_name(name: str) -> None:
    get_tracer().set_thread_name(name)


def save_trace(path: str = "trace.json") -> None:
    get_tracer().save(path)
