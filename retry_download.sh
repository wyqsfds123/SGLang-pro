#!/bin/bash
# Auto-retry download script for Qwen3-32B
# Resumes partial downloads, retries on failure

source ~/Documents/sglang_env/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://127.0.0.1:8888
export https_proxy=http://127.0.0.1:8888

EXPECTED_SIZE=66000000000  # ~64GB
MODEL_DIR=/home/fnl/models/Qwen3-32B

while true; do
    CURRENT_SIZE=$(du -sb "$MODEL_DIR" 2>/dev/null | awk '{print $1}')

    if [ "$CURRENT_SIZE" -ge "$EXPECTED_SIZE" ] 2>/dev/null; then
        SHARDS=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
        if [ "$SHARDS" -ge 17 ]; then
            echo "[$(date '+%H:%M:%S')] Download COMPLETE! $SHARDS shards, $(du -sh $MODEL_DIR | awk '{print $1}')"
            exit 0
        fi
    fi

    echo "[$(date '+%H:%M:%S')] Current: $(du -sh $MODEL_DIR 2>/dev/null | awk '{print $1}') | Starting download attempt..."

    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B', local_dir='$MODEL_DIR')
" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Download finished successfully!"
        exit 0
    fi

    echo "[$(date '+%H:%M:%S')] Download failed (exit $EXIT_CODE), waiting 30s before retry..."
    sleep 30
done
