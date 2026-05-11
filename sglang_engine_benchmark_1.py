#!/usr/bin/env python3
"""Benchmark KV cache behavior through sglang_engine_server.py.

Prerequisite:
    cd /home/fnl/Documents
    source sglang_env/bin/activate
    uvicorn sglang_engine_server:app --host 0.0.0.0 --port 8001 --workers 1

Run:
    cd /home/fnl/Documents
    source sglang_env/bin/activate
    python sglang_engine_kv_benchmark.py

This script does not load the model. It calls the persistent Engine server and
checks whether repeated/prefix-shared prompts report cached_tokens.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_SERVER = "http://127.0.0.1:8001"


LONG_SYSTEM_PROMPT = """你是一个专业的技术顾问，需要严格遵循以下规则来回答问题：

【规则1】回答必须结构化，使用编号列表。
【规则2】每个要点控制在两句话以内。
【规则3】如果涉及代码，必须使用 Markdown 代码块。
【规则4】优先给出结论，再解释原因。
【规则5】对不确定的内容要明确标注"待验证"。

【知识背景】
KV Cache（键值缓存）是大语言模型推理优化的核心技术之一。在 Transformer 的自注意力机制中，每个 token 都需要与前面所有 token 计算注意力分数。生成第 N 个 token 时，需要重新计算前 N-1 个 token 的 Key 和 Value 向量。KV Cache 通过缓存已计算过的 Key-Value 对，避免重复计算，从而大幅加速推理。

KV Cache 的内存占用与序列长度、层数、注意力头数和头维度成正比。对于大型模型，KV Cache 可能占用数 GB 的显存。RadixAttention 将 KV Cache 组织为 Radix Tree 结构，使得不同请求之间可以高效共享相同的 KV 前缀。当一个新请求与已缓存的请求共享前缀时，可以直接复用对应的 KV Cache，跳过 prefill 阶段的部分计算。

这对多轮对话、长 system prompt 场景尤其有效：system prompt 通常在所有请求中保持不变，只需计算一次即可反复复用。RadixAttention 还使用 LRU 策略自动管理缓存淘汰，在内存不足时优先淘汰最久未使用的缓存。

在实际部署中，RadixAttention 可以将带有长 system prompt 的请求的 TTFT（Time To First Token）降低数倍。缓存命中率越高，加速效果越明显。这使得 SGLang 在处理大量相似请求时具有显著的性能优势。

请严格遵循以上所有规则来回答用户的问题。"""


@dataclass
class Result:
    label: str
    ttft: float
    total: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    text: str

    @property
    def cache_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens


def build_prompt(system_msg: str, user_msg: str) -> str:
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def check_server(server: str, timeout: float) -> None:
    try:
        response = requests.get(f"{server}/health", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"无法连接 Engine Server: {server}\n"
            "请先启动:\n"
            "  cd /home/fnl/Documents\n"
            "  source sglang_env/bin/activate\n"
            "  uvicorn sglang_engine_server:app --host 0.0.0.0 --port 8001 --workers 1\n"
            f"原始错误: {exc}"
        ) from exc

    data = response.json()
    if data.get("status") != "ok":
        raise SystemExit(f"Engine Server 尚未就绪: {data}")

    print("Engine Server 已就绪")
    print(f"  地址:  {server}")
    print(f"  模型:  {data.get('model_path')}")
    print(f"  引擎:  {data.get('engine_host')}:{data.get('engine_port')}")
    print(f"  运行:  {data.get('uptime_seconds')}s")


def call_generate(
    server: str,
    prompt: str,
    label: str,
    max_new_tokens: int,
    timeout: float,
) -> Result:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1,
        "return_meta": True,
    }

    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")

    wall_start = time.perf_counter()
    response = requests.post(f"{server}/generate", json=payload, timeout=timeout)
    wall_total = time.perf_counter() - wall_start
    response.raise_for_status()
    data = response.json()

    timings = data.get("timings") or {}
    meta = data.get("meta_info") or {}
    text = data.get("text") or ""

    ttft = float(timings.get("ttft_seconds") or wall_total)
    total = float(timings.get("total_seconds") or wall_total)
    prompt_tokens = int(meta.get("prompt_tokens") or 0)
    cached_tokens = int(meta.get("cached_tokens") or 0)
    completion_tokens = int(meta.get("completion_tokens") or 0)

    result = Result(
        label=label,
        ttft=ttft,
        total=total,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        text=text,
    )

    print(f"  回答预览:      {text.strip()[:80]}...")
    print(f"  TTFT:          {result.ttft:.3f}s")
    print(f"  服务总耗时:    {result.total:.3f}s")
    print(f"  客户端墙钟:    {wall_total:.3f}s")
    print(f"  prompt_tokens: {result.prompt_tokens}")
    print(
        f"  cached_tokens: {result.cached_tokens} "
        f"({result.cache_ratio * 100:.1f}%)"
    )
    print(f"  completion:    {result.completion_tokens}")

    return result


def pct(cached: int, prompt: int) -> str:
    if prompt <= 0:
        return "N/A"
    return f"{cached / prompt * 100:.1f}%"


def print_summary(results: list[Result]) -> None:
    print(f"\n{'=' * 68}")
    print("  汇总报告")
    print(f"{'=' * 68}")
    print(
        f"  {'测试':28s} {'TTFT':>8s} {'总耗时':>8s} "
        f"{'prompt':>8s} {'cached':>8s} {'命中率':>8s}"
    )
    print(f"  {'-' * 74}")
    for item in results:
        print(
            f"  {item.label:28s} {item.ttft:>7.3f}s {item.total:>7.3f}s "
            f"{item.prompt_tokens:>8d} {item.cached_tokens:>8d} "
            f"{pct(item.cached_tokens, item.prompt_tokens):>8s}"
        )


def print_verdict(cold: Result, same: Result, prefix: Result) -> None:
    print(f"\n{'=' * 68}")
    print("  验证结论")
    print(f"{'=' * 68}")

    if same.cached_tokens > cold.cached_tokens:
        speedup = cold.ttft / same.ttft if same.ttft > 0 else 0.0
        print("  重复 prompt KV Cache: 检测到缓存命中")
        print(
            f"  - 冷请求 cached={cold.cached_tokens}/{cold.prompt_tokens}, "
            f"TTFT={cold.ttft:.3f}s"
        )
        print(
            f"  - 重复请求 cached={same.cached_tokens}/{same.prompt_tokens}, "
            f"TTFT={same.ttft:.3f}s"
        )
        print(f"  - TTFT 对比: {speedup:.2f}x")
    else:
        print("  重复 prompt KV Cache: 未检测到明显 cached_tokens 增长")

    if prefix.cached_tokens > 0:
        print("  Prefix Cache 复用: 检测到缓存命中")
        print(
            f"  - prefix 请求 cached={prefix.cached_tokens}/"
            f"{prefix.prompt_tokens} ({pct(prefix.cached_tokens, prefix.prompt_tokens)})"
        )
    else:
        print("  Prefix Cache 复用: 未检测到 cached_tokens")

    print("\n  说明: 如果服务已经运行过其他请求，冷请求也可能命中历史缓存。")
    print("  想观察更干净的冷启动对比，可以重启 sglang_engine_server.py 后再运行本脚本。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify SGLang KV cache through the persistent Engine server."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Engine server URL")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = args.server.rstrip("/")

    print("=" * 68)
    print("  SGLang Engine Server KV Cache 验证")
    print("  接口: /generate")
    print("=" * 68)

    check_server(server, args.timeout)

    prompt_a = build_prompt(LONG_SYSTEM_PROMPT, "解释什么是机器学习")
    prompt_b = build_prompt(LONG_SYSTEM_PROMPT, "解释什么是深度学习")
    flush_prompt = build_prompt("你是一个数学家", "2+2等于几？")

    print("\n  [第0步] 发送一个无关请求，降低历史 prefix 干扰。")
    call_generate(
        server,
        flush_prompt,
        "预热/无关请求",
        max_new_tokens=16,
        timeout=args.timeout,
    )
    time.sleep(args.sleep)

    cold = call_generate(
        server,
        prompt_a,
        "测试1: 长 system prompt + 问题A",
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )
    time.sleep(args.sleep)

    same = call_generate(
        server,
        prompt_a,
        "测试2: 完全相同 prompt",
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )
    time.sleep(args.sleep)

    prefix = call_generate(
        server,
        prompt_b,
        "测试3: 相同 system prompt + 问题B",
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )

    print_summary([cold, same, prefix])
    print_verdict(cold, same, prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())

