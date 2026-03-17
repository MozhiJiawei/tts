"""
Chrome Tracing 多线程打点示例

运行后会在当前目录生成 trace.json，在 Chrome 地址栏打开 chrome://tracing ，
点击 "Load" 加载该文件即可查看时间线。

本示例演示：
- span() 上下文管理器：自动记录一段代码的起止
- begin()/end() 手动打点
- instant() 瞬时事件（无时长）
- 多线程下各线程的调度与重叠关系
- set_thread_name() 在时间线中显示线程名称
"""

import random
import threading
import time

from chrome_tracing import (
    get_tracer,
    reset_tracer,
    span,
    begin,
    end,
    instant,
    set_thread_name,
    save_trace,
)


def worker(name: str, task_count: int) -> None:
    """模拟工作线程：执行多轮「准备 -> 计算 -> 写回」任务。"""
    set_thread_name(name)

    for i in range(task_count):
        # 方式1：用 span 自动记录整段区间
        with span(f"任务-{i}", cat="worker", args={"thread": name, "task_id": i}):
            # 准备阶段（可再打子区间）
            begin("准备数据", cat="worker")
            time.sleep(random.uniform(0.02, 0.08))
            end("准备数据", cat="worker")

            # 计算阶段
            begin("计算", cat="worker")
            time.sleep(random.uniform(0.05, 0.15))
            end("计算", cat="worker")

            # 瞬时事件：标记关键点（无时长）
            instant("检查点", cat="worker", args={"task": i})

            # 写回阶段
            with span("写回结果", cat="worker"):
                time.sleep(random.uniform(0.02, 0.06))


def main() -> None:
    reset_tracer()
    tracer = get_tracer()
    tracer.set_process_name("多线程调度示例")

    instant("程序启动", cat="main")

    threads = []
    for i in range(4):
        t = threading.Thread(
            target=worker,
            args=(f"Worker-{i}", 3),
            name=f"Worker-{i}",
        )
        threads.append(t)

    begin("启动所有线程", cat="main")
    for t in threads:
        t.start()
    end("启动所有线程", cat="main")

    instant("所有线程已启动", cat="main")

    for t in threads:
        t.join()

    instant("所有线程结束", cat="main")

    out_path = "trace.json"
    save_trace(out_path)
    print(f"Trace 已保存到: {out_path}")
    print("在 Chrome 中打开 chrome://tracing ，点击 Load 加载该文件即可查看时间线。")


if __name__ == "__main__":
    main()
