#!/usr/bin/env python3
"""Benchmark SGLang RadixAttention KV Cache over HTTP.

使用 SGLang 原生 /generate 端点（streaming 模式），精确测量 TTFT，
并通过 meta_info.cached_tokens 验证缓存命中情况。

测试流程：
  0. 发送无关请求冲刷旧缓存
  1. 冷启动请求（长 system prompt + 问题A）→ 应无缓存
  2. 完全相同请求 → KV Cache 应全部命中
  3. 不同问题但相同 system prompt → prefix cache 复用

前提：SGLang 服务已启动在 localhost:30000
  source ~/Documents/sglang_env/bin/activate
  python -m sglang.launch_server --model-path /home/fnl/models/Qwen3-32B \
    --host 0.0.0.0 --port 30000 --mem-fraction-static 0.80
"""

import time
import json
import requests

GENERATE_URL = "http://localhost:30000/generate"

# 超长 system prompt（约 800 token），让缓存效果在 TTFT 上可观测
LONG_SYSTEM_PROMPT = """你是一个专业的技术顾问，需要严格遵循以下规则来回答问题：

【规则1】回答必须结构化，使用编号列表。
【规则2】每个要点控制在两句话以内。
【规则3】如果涉及代码，必须使用 Markdown 代码块。
【规则4】优先给出结论，再解释原因。
【规则5】对不确定的内容要明确标注"待验证"。

【知识背景】
KV Cache（键值缓存）是大语言模型推理优化的核心技术之一。在 Transformer 的自注意力机制中，每个 token 都需要与前面所有 token 计算注意力分数。生成第 N 个 token 时，需要重新计算前 N-1 个 token 的 Key 和 Value 向量。KV Cache 通过缓存已计算过的 Key-Value 对，避免重复计算，从而大幅加速推理。

KV Cache 的内存占用与序列长度、层数、注意力头数和头维度成正比。对于大型模型（如 70B 参数），KV Cache 可能占用数 GB 的显存。RadixAttention 是 SGLang 提出的创新技术，它将 KV Cache 组织为 Radix Tree（基数树）结构，使得不同请求之间可以高效共享相同的 KV 前缀。当一个新请求与已缓存的请求共享前缀时，可以直接复用对应的 KV Cache，跳过 prefill 阶段的部分计算。

这对多轮对话、长 system prompt 场景尤其有效：system prompt 通常在所有请求中保持不变，只需计算一次即可反复复用。RadixAttention 还使用 LRU（最近最少使用）策略自动管理缓存淘汰，在内存不足时优先淘汰最久未使用的缓存。

在实际部署中，RadixAttention 可以将带有长 system prompt 的请求的 TTFT（Time To First Token）降低数倍。缓存命中率越高，加速效果越明显。这使得 SGLang 在处理大量相似请求时具有显著的性能优势。

请严格遵循以上所有规则来回答用户的问题。"""


def build_prompt(system_msg, user_msg):
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def call_stream(prompt_text, label, max_new_tokens=50):
    """streaming 模式调用 /generate，精确测量 TTFT，提取 cached_tokens"""
    payload = {
        "text": prompt_text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        "stream": True,
    }

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    start = time.perf_counter()
    ttft = None
    full_text = ""
    meta_info = {}

    r = requests.post(GENERATE_URL, json=payload, stream=True)
    for line in r.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        text = chunk.get("text", "")
        if text and ttft is None:
            ttft = time.perf_counter() - start
        full_text += text

        mi = chunk.get("meta_info", {})
        if mi:
            meta_info = mi

    total = time.perf_counter() - start
    if ttft is None:
        ttft = total

    pt = meta_info.get("prompt_tokens", 0)
    ct = meta_info.get("cached_tokens", 0)
    comp = meta_info.get("completion_tokens", 0)
    ratio = ct / pt * 100 if pt > 0 else 0

    print(f"  回答:          {full_text.strip()[:60]}...")
    print(f"  TTFT:          {ttft:.3f}s")
    print(f"  总耗时:        {total:.3f}s")
    print(f"  prompt_tokens: {pt}")
    print(f"  cached_tokens: {ct} ({ratio:.1f}%)")
    print(f"  completion:    {comp}")

    return ttft, total, pt, ct


def main():
    print("=" * 60)
    print("  SGLang RadixAttention KV Cache 验证测试")
    print("  端点: /generate (streaming, 返回 cached_tokens)")
    print("  模型: Qwen3-32B | 服务: http://localhost:30000")
    print("=" * 60)

    # 构造测试 prompt
    prompt_a = build_prompt(LONG_SYSTEM_PROMPT, "解释什么是机器学习")
    prompt_b = build_prompt(LONG_SYSTEM_PROMPT, "解释什么是深度学习")
    flush_prompt = build_prompt("你是一个数学家", "2+2等于几")

    # ===== 第 0 步：冲刷旧缓存 =====
    print("\n  [第0步] 发送无关请求，冲刷可能残留的旧缓存...")
    call_stream(flush_prompt, "冲刷请求（无关内容）", max_new_tokens=20)
    time.sleep(1)

    # ===== 测试 1：冷启动 =====
    print("\n  >>> 以下请求的缓存应为空 <<<")
    t1_ttft, t1_total, pt1, ct1 = call_stream(
        prompt_a, "测试1: 冷启动（无缓存）"
    )
    time.sleep(1)

    # ===== 测试 2：完全相同请求 =====
    t2_ttft, t2_total, pt2, ct2 = call_stream(
        prompt_a, "测试2: 完全相同请求（KV Cache 应全部命中）"
    )
    time.sleep(1)

    # ===== 测试 3：不同问题，相同 system prompt =====
    t3_ttft, t3_total, pt3, ct3 = call_stream(
        prompt_b, "测试3: 不同问题，system prefix 复用"
    )

    # ===== 汇总报告 =====
    def pct(c, p):
        return f"{c/p*100:.1f}%" if p > 0 else "N/A"

    results = [
        ("1.冷启动", t1_ttft, t1_total, pt1, ct1),
        ("2.完全相同(全命中)", t2_ttft, t2_total, pt2, ct2),
        ("3.换问题(prefix复用)", t3_ttft, t3_total, pt3, ct3),
    ]

    print(f"\n{'='*60}")
    print(f"  汇总报告")
    print(f"{'='*60}")
    print(f"  {'测试':26s} {'TTFT':>8s} {'总耗时':>8s} {'prompt':>7s} {'cached':>7s} {'命中率':>7s}")
    print(f"  {'-'*63}")
    for label, ttft, total, pt, ct in results:
        print(f"  {label:26s} {ttft:>7.3f}s {total:>7.3f}s {pt:>7d} {ct:>7d} {pct(ct,pt):>7s}")

    # ===== 验证结论 =====
    print(f"\n{'='*60}")
    print(f"  验证结论")
    print(f"{'='*60}")

    # 判断1：KV Cache 全量命中
    if ct1 <= pt1 * 0.1 and ct2 >= pt2 * 0.9:
        speedup = t1_ttft / t2_ttft if t2_ttft > 0 else 0
        print(f"  KV Cache 全量命中验证: 通过")
        print(f"  - 测试1 (冷启动): {ct1}/{pt1} 缓存, TTFT={t1_ttft:.3f}s")
        print(f"  - 测试2 (全命中): {ct2}/{pt2} 缓存, TTFT={t2_ttft:.3f}s")
        print(f"  - TTFT 加速: {speedup:.1f}x")
    elif ct2 > ct1:
        print(f"  KV Cache 部分生效")
        print(f"  - cached_tokens: 测试1={ct1}, 测试2={ct2}")
    else:
        print(f"  KV Cache 未检测到效果 (测试1={ct1}, 测试2={ct2})")

    # 判断2：Prefix Cache 复用
    if ct3 >= pt3 * 0.5:
        print(f"  Prefix Cache 复用验证: 通过")
        print(f"  - 测试3 缓存 {ct3}/{pt3} tokens ({pct(ct3,pt3)})")
        uncached = pt3 - ct3
        print(f"  - 未缓存 {uncached} tokens (问题部分: '机器学习' vs '深度学习')")
    else:
        print(f"  Prefix Cache 复用未检测到 (测试3 cached={ct3}/{pt3})")

    print()


if __name__ == "__main__":
    main()
