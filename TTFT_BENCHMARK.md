# TTFT Benchmark Report — Qwen3-32B on NVIDIA GB10 (Grace-Blackwell)

> 测试环境: Lenovo ThinkStation PGX | Ubuntu 24.04 LTS (aarch64) | NVIDIA GB10 GPU | CUDA 13.0 | Driver 580.95.05
>
> 模型: Qwen3-32B (FP16, ~62 GB, 17 safetensors shards)
>
> 推理框架: SGLang (源码安装, 含 RadixAttention 前缀缓存)

---

## 1. 什么是 TTFT

**TTFT (Time To First Token)** 是大语言模型推理服务中最关键的性能指标之一，定义为**从用户发送请求到模型输出第一个有效 token 的时间**。

### 1.1 为什么 TTFT 重要

LLM 推理分为两个阶段：

| 阶段 | 做什么 | 耗时特点 |
|------|--------|----------|
| **Prefill** | 处理完整输入 prompt，计算所有 token 的 Key-Value 向量（KV Cache） | 与输入长度成正比，是 TTFT 的主要组成部分 |
| **Decode** | 基于 KV Cache 逐个自回归生成后续 token | 每个 token 耗时相对稳定 |

TTFT 主要反映 **Prefill 阶段**的性能。在以下场景中 TTFT 尤为关键：

- **长 system prompt**：如 RAG 应用中注入大量检索上下文，prompt 可达数千 token
- **多轮对话**：历史对话每轮累积，prefill 计算量随轮次增长
- **并发服务**：多个用户同时请求，GPU 资源竞争影响 TTFT

### 1.2 KV Cache 与 TTFT 的关系

SGLang 的 **RadixAttention** 技术通过基数树（Radix Tree）管理 KV Cache，使不同请求可共享相同的 KV 前缀：

- **缓存未命中（Cold）**：完整计算所有 prompt token 的 KV，TTFT = 完整 prefill 时间
- **缓存全命中（Full Hit）**：所有 prompt token 的 KV 已缓存，跳过 prefill，TTFT 接近零
- **前缀部分命中（Prefix Hit）**：system prompt 部分复用缓存，仅需计算差异部分

> 核心公式: `TTFT = f(prompt_tokens - cached_tokens)`，缓存命中的 token 越多，TTFT 越低

---

## 2. 测试指标体系

| 指标 | 全称 | 单位 | 含义 |
|------|------|------|------|
| **TTFT** | Time To First Token | ms | 首个 token 延迟，核心指标 |
| **TPOT** | Time Per Output Token | ms | 每个 decode token 平均耗时（不含首 token） |
| **ITL** | Inter-Token Latency | ms | 相邻 token 间延迟，反映生成流畅度 |
| **E2E Latency** | End-to-End Latency | ms | 从请求发出到生成完成的端到端总时间 |
| **cached_tokens** | — | 个 | 命中 KV Cache 的 token 数 |
| **cache_hit_ratio** | — | % | `cached_tokens / prompt_tokens × 100%` |
| **Throughput** | Request Throughput | req/s | 单位时间完成的请求数 |

---

## 3. Benchmark 脚本说明

本项目包含 4 个层次的 TTFT 测试工具，从单请求验证到并发压测，覆盖不同场景。

### 3.1 `test_sglang_cache.py` — 进程内 KV Cache 验证

**测试模式**: 进程内直接使用 `sglang.Engine` API（无 HTTP 开销）

**测试流程**:
1. 进程内启动 SGLang Engine（加载 Qwen3-32B）
2. 发送首次请求（冷启动，无缓存）
3. 发送完全相同请求（KV Cache 全命中）
4. 发送不同问题但相同 system prompt（prefix cache 复用）

**输出示例**:

```
  第1次请求 - 冷启动，无缓存
  TTFT:          2.145s
  prompt tokens: 52  |  cached: 0  |  completion: 128

  第2次请求 - 完全相同，KV Cache 命中
  TTFT:          0.082s
  prompt tokens: 52  |  cached: 52  |  completion: 128

  第3次请求 - 不同问题，system prefix 复用
  TTFT:          0.610s
  prompt tokens: 54  |  cached: 28  |  completion: 130

  TTFT 加速: 26.2x
```

**适合场景**: 开发调试、验证 RadixAttention 是否正常工作

### 3.2 `sglang_kv_benchmark_http.py` — HTTP Streaming KV Cache 验证

**测试模式**: 通过 SGLang 原生 `/generate` 端点，streaming 模式精确测量 TTFT

**测试流程**:
1. 发送无关请求冲刷旧缓存
2. 冷启动请求（~800 token 长 system prompt + 问题A）
3. 完全相同请求 → KV Cache 全命中
4. 不同问题但相同 system prompt → prefix cache 复用

**关键设计**: 使用约 800 token 的长 system prompt，使缓存效果在 TTFT 上有明显可观测差异。

**输出示例**:

```
  汇总报告
  ─────────────────────────────────────────────────────────────
  测试                          TTFT    总耗时  prompt  cached  命中率
  1.冷启动                     3.421s  12.305s     812       0    0.0%
  2.完全相同(全命中)            0.098s   8.712s     812     812  100.0%
  3.换问题(prefix复用)          1.240s  10.110s     818     780   95.4%

  验证结论
  KV Cache 全量命中验证: 通过
  - TTFT 加速: 34.9x
  Prefix Cache 复用验证: 通过
  - prefix 请求缓存 780/818 tokens (95.4%)
```

**适合场景**: 验证 HTTP 服务层的 KV Cache 效果，贴近真实部署场景

### 3.3 `sglang_engine_benchmark_1.py` — Engine Server KV Cache 验证

**测试模式**: 通过自建 FastAPI Engine Server（端口 8001）的 REST 接口测试

**测试流程**: 与 3.2 相同，但通过 `sglang_engine_server.py` 的 `/generate` 端点调用

**额外输出**: 服务端精确计时（`ttft_seconds`、`total_seconds`）和客户端墙钟时间对比

**适合场景**: 验证自定义封装 Engine Server 的缓存行为

### 3.4 `bench_warm_cache.py` — 前缀复用率压测（SGLang 自带）

**测试模式**: 精确控制 shared-prefix 比例（0%~99%），多请求并发压测

**关键参数**:

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `--num-prompts` | 并发请求数 | 64 |
| `--total-tokens` | 每个 prompt 的总 token 数 | 70000 |
| `--output-len` | 每个 prompt 的输出 token 数 | 200 |
| `--pcts` | 共享前缀比例列表 | 0,10,20,50,80,90,95,99 |

**输出**: 对每个前缀复用比例输出完整的 TTFT 统计（均值/中位/P90/P99），可绘制「前缀复用率 vs TTFT」曲线。

**适合场景**: 学术研究、论文实验数据、性能报告

---

## 4. 快速运行指南

### 4.1 启动 SGLang 服务

```bash
source /home/fnl/Documents/sglang_env/bin/activate
python -m sglang.launch_server \
    --model-path /home/fnl/models/Qwen3-32B \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.80
```

### 4.2 运行测试

```bash
# 测试 A: HTTP streaming TTFT 测试（推荐首选）
python sglang_kv_benchmark_http.py

# 测试 B: 通用在线服务压测（完整 TTFT 统计）
python -m sglang.bench_serving \
    --backend sglang --base-url http://127.0.0.1:30000 \
    --model /home/fnl/models/Qwen3-32B \
    --dataset-name random --num-prompts 64 \
    --random-input 1024 --random-output 256

# 测试 C: 精确控制前缀复用率的 TTFT 压测
python /home/fnl/Documents/sglang/benchmark/hicache/bench_warm_cache.py \
    --model /home/fnl/models/Qwen3-32B \
    --base-url http://127.0.0.1:30000 \
    --num-prompts 32 --total-tokens 2000 --output-len 100 \
    --pcts 0,50,90,99
```

---

## 5. 结果解读

### 5.1 如何判断 KV Cache 是否生效

| 观察项 | 缓存未生效 | 缓存已生效 |
|--------|-----------|-----------|
| `cached_tokens` | 始终为 0 或接近 0 | 重复请求时 > 90% prompt_tokens |
| TTFT（重复请求） | 与首次请求相近 | 显著下降（通常 > 5x 加速） |
| TTFT（prefix 复用） | 与全新请求相近 | 下降幅度 ≈ 缓存占比 |

### 5.2 影响因素

- **输入长度**: prompt 越长，prefill 越耗时，缓存收益越大
- **GPU 显存**: `mem-fraction-static` 决定可用于 KV Cache 的显存比例
- **并发度**: 高并发时 GPU 资源竞争，TTFT 可能上升
- **前缀复用率**: 共享前缀比例越高，缓存命中越多，TTFT 越低

---

## 6. 项目架构

```
SGLang-pro/
├── README.md                        # 项目总览
├── TTFT_BENCHMARK.md                # 本文档 — TTFT 测试指标说明
├── sglang_engine_server.py          # FastAPI 持久化 Engine 服务
├── sglang_engine_benchmark_1.py     # Engine Server KV Cache 验证
├── sglang_kv_benchmark_http.py      # HTTP Streaming TTFT 测试
├── test_sglang_cache.py             # 进程内 KV Cache 测试
├── retry_download.sh                # 模型下载脚本（自动重试）
├── requirements.txt                 # Python 依赖
├── requirements-lock.txt            # 冻结依赖版本
└── env.example                      # 环境变量模板
```

关联仓库:
- **SGLang-pro** (本项目): https://github.com/wyqsfds123/SGLang-pro
- **SGLang Fork** (推理框架源码): https://github.com/wyqsfds123/sglang
