
1#!/usr/bin/env python3
"""SGLang 可编程模式 - RadixAttention KV Cache 效果演示

直接使用 sglang.Engine API，无需启动 HTTP 服务。
Engine 在进程内启动推理引擎，通过 ZMQ IPC 通信。
"""
import time
from sglang.srt.entrypoints.engine import Engine


def main():
    # ====== 1. 启动引擎 ======
    print("正在启动 SGLang Engine（进程内模式）...")
    engine = Engine(
        model_path="/home/fnl/models/Qwen3-32B",
        mem_fraction_static=0.80,
        host="0.0.0.0",
        port=30001,
    )
    print("引擎就绪!\n")

    # ====== 2. 定义测试函数 ======
    def measure_request(prompt_text, label):
        """发一次请求，测量 TTFT 和总耗时"""
        user_msg = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        print(f"{'='*55}")
        print(f"  [{label}]")
        print(f"  问题: {user_msg}")
        print(f"{'='*55}")

        ttft = None
        full_text = ""
        cached_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        start = time.perf_counter()
        for chunk in engine.generate(
            prompt=prompt_text,
            sampling_params={"max_new_tokens": 200, "temperature": 0},
            stream=True,
        ):
            delta = chunk.get("text", "")
            meta_info = chunk.get("meta_info", {})

            if delta and ttft is None:
                ttft = time.perf_counter() - start

            full_text += delta
            cached_tokens = meta_info.get("cached_tokens", cached_tokens)
            prompt_tokens = meta_info.get("prompt_tokens", prompt_tokens)
            completion_tokens = meta_info.get("completion_tokens", completion_tokens)

            if chunk.get("finish_reason") is not None:
                break

        if ttft is None:
            ttft = time.perf_counter() - start
        total_time = time.perf_counter() - start

        print(f"  回答: {full_text.strip()[:200]}")
        print(f"  TTFT:          {ttft:.3f}s")
        print(f"  总耗时:        {total_time:.3f}s")
        print(f"  prompt tokens: {prompt_tokens}  |  cached: {cached_tokens}  |  completion: {completion_tokens}")

        return ttft, total_time, cached_tokens

    # Engine.generate 用 chat 格式需要传 text prompt
    SYSTEM_PROMPT = "你是一个专业的中文天气助手。回答控制在2句话以内。不要输出思考过程，直接回答。"

    prompt1 = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n今天天气怎么样？我在北京。<|im_end|>\n<|im_start|>assistant\n"
    prompt2 = prompt1  # 完全相同
    prompt3 = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n今天天气怎么样？我在上海。<|im_end|>\n<|im_start|>assistant\n"

    # ====== 3. 三次请求对比 ======

    # 第一次：冷启动
    ttft1, total1, cached1 = measure_request(prompt1, "第1次请求 - 冷启动，无缓存")

    time.sleep(1)

    # 第二次：完全相同的 prompt，KV Cache 应全部命中
    ttft2, total2, cached2 = measure_request(prompt2, "第2次请求 - 完全相同，KV Cache 命中")

    time.sleep(1)

    # 第三次：换问题但 system prompt 相同，prefix cache 复用
    ttft3, total3, cached3 = measure_request(prompt3, "第3次请求 - 不同问题，system prefix 复用")

    # ====== 4. 对比报告 ======
    print(f"\n{'='*55}")
    print(f"  SGLang RadixAttention 缓存效果对比")
    print(f"{'='*55}")
    print(f"  {'':22s} {'TTFT':>8s} {'总耗时':>8s} {'缓存token':>10s}")
    print(f"  {'第1次(冷启动)':22s} {ttft1:>7.3f}s {total1:>7.3f}s {cached1:>10d}")
    print(f"  {'第2次(完全命中)':22s} {ttft2:>7.3f}s {total2:>7.3f}s {cached2:>10d}")
    print(f"  {'第3次(prefix复用)':22s} {ttft3:>7.3f}s {total3:>7.3f}s {cached3:>10d}")
    print()

    if ttft2 < ttft1:
        print(f"  第2次 vs 第1次: TTFT 加速 {ttft1/ttft2:.1f}x")
        print(f"    -> KV Cache 命中 {cached2} tokens，跳过 prefill 计算")
    if cached3 > 0:
        print(f"  第3次: 复用了 {cached3} tokens 的 prefix cache (system prompt)")
    print(f"\n  原理: SGLang RadixAttention 以 radix tree 管理 KV Cache，")
    print(f"  相同前缀的 KV 不需要重新计算，直接复用。")

    # ====== 5. 关闭引擎 ======
    print(f"\n正在关闭引擎...")
    engine.shutdown()
    print("引擎已关闭。")


if __name__ == "__main__":
    main()
