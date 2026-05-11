# SGLang Pro

SGLang inference engine server with KV Cache benchmarking tools for Qwen3-32B on NVIDIA DGX Spark (GB10 Grace-Blackwell).

## Files

| File | Description |
|------|-------------|
| `sglang_engine_server.py` | FastAPI server wrapping SGLang Engine with `/generate` and `/generate_stream` endpoints |
| `sglang_engine_benchmark_1.py` | KV Cache hit-rate benchmark via the custom engine server (port 8001) |
| `sglang_kv_benchmark_http.py` | KV Cache benchmark via SGLang native HTTP server (port 30000) |
| `test_sglang_cache.py` | In-process SGLang Engine API cache test |
| `retry_download.sh` | Auto-retry Qwen3-32B model download script |

## Setup

```bash
# Install SGLang from source
git clone https://github.com/sgl-project/sglang.git
cd sglang && pip install -e "python[all]"

# Install dependencies
pip install -r requirements.txt

# Download model
bash retry_download.sh
```

## Usage

### Start the engine server
```bash
export SGL_MODEL_PATH=/path/to/Qwen3-32B
uvicorn sglang_engine_server:app --host 0.0.0.0 --port 8001
```

### Run benchmarks
```bash
python sglang_engine_benchmark_1.py
python sglang_kv_benchmark_http.py
python test_sglang_cache.py
```

## Hardware

- NVIDIA DGX Spark (GB10 Grace-Blackwell)
- CUDA 13.0, Driver 580.95.05
- Ubuntu 24.04 LTS (arm64)
