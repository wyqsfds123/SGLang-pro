#!/usr/bin/env python3
"""Persistent SGLang Engine server.

Run with:
    cd /home/fnl/Documents
    source sglang_env/bin/activate
    uvicorn sglang_engine_server:app --host 0.0.0.0 --port 8001 --workers 1

The FastAPI process owns one SGLang Engine instance. Other scripts should call
this server over HTTP instead of creating their own Engine, so the model stays
loaded in GPU memory across requests.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from sglang.srt.entrypoints.engine import Engine


logging.basicConfig(
    level=os.getenv("SGL_ENGINE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sglang_engine_server")


MODEL_PATH = os.getenv("SGL_MODEL_PATH", "/home/fnl/models/Qwen3-32B")
ENGINE_HOST = os.getenv("SGL_ENGINE_HOST", "0.0.0.0")
ENGINE_PORT = int(os.getenv("SGL_ENGINE_PORT", "30001"))
MEM_FRACTION_STATIC = float(os.getenv("SGL_MEM_FRACTION_STATIC", "0.80"))

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("SGL_DEFAULT_MAX_NEW_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.getenv("SGL_DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("SGL_DEFAULT_TOP_P", "0.95"))


engine: Engine | None = None
started_at: float | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    prompt: str | None = Field(
        default=None,
        description="Raw prompt. If set, messages/system_prompt are ignored.",
    )
    messages: list[ChatMessage] | None = Field(
        default=None,
        description="Chat messages converted to Qwen chat-template text.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt used only when prompt is not set.",
    )
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=8192)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    stop: list[str] | None = None
    return_meta: bool = True

    @model_validator(mode="after")
    def require_prompt_or_messages(self) -> "GenerateRequest":
        if not self.prompt and not self.messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        return self


class GenerateResponse(BaseModel):
    text: str
    timings: dict[str, float]
    meta_info: dict[str, Any] | None = None


def build_chat_prompt(messages: list[ChatMessage], system_prompt: str | None) -> str:
    parts: list[str] = []

    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")

    for message in messages:
        parts.append(
            f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        )

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def request_to_prompt(req: GenerateRequest) -> str:
    if req.prompt:
        return req.prompt
    assert req.messages is not None
    return build_chat_prompt(req.messages, req.system_prompt)


def request_to_sampling_params(req: GenerateRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }
    if req.stop:
        params["stop"] = req.stop
    return params


def require_engine() -> Engine:
    if engine is None:
        raise HTTPException(status_code=503, detail="SGLang Engine is not ready.")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, started_at

    logger.info("Starting SGLang Engine")
    logger.info(
        "model_path=%s host=%s port=%s mem_fraction_static=%.2f",
        MODEL_PATH,
        ENGINE_HOST,
        ENGINE_PORT,
        MEM_FRACTION_STATIC,
    )

    engine = Engine(
        model_path=MODEL_PATH,
        host=ENGINE_HOST,
        port=ENGINE_PORT,
        mem_fraction_static=MEM_FRACTION_STATIC,
    )
    started_at = time.time()
    logger.info("SGLang Engine is ready")

    try:
        yield
    finally:
        if engine is not None:
            logger.info("Shutting down SGLang Engine")
            engine.shutdown()
            engine = None
            logger.info("SGLang Engine has stopped")


app = FastAPI(
    title="Persistent SGLang Engine Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if engine is not None else "starting",
        "model_path": MODEL_PATH,
        "engine_host": ENGINE_HOST,
        "engine_port": ENGINE_PORT,
        "uptime_seconds": round(time.time() - started_at, 3)
        if started_at is not None
        else 0,
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "owned_by": "local-sglang-engine",
            }
        ],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    runtime = require_engine()
    prompt = request_to_prompt(req)
    sampling_params = request_to_sampling_params(req)

    start = time.perf_counter()
    ttft: float | None = None
    full_text = ""
    meta_info: dict[str, Any] = {}

    try:
        generator = await runtime.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
            stream=True,
        )
        async for chunk in generator:
            text = chunk.get("text", "")
            if text and ttft is None:
                ttft = time.perf_counter() - start
            full_text += text

            chunk_meta = chunk.get("meta_info", {})
            if chunk_meta:
                meta_info = chunk_meta

            if chunk.get("finish_reason") is not None:
                break
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total = time.perf_counter() - start
    return GenerateResponse(
        text=full_text,
        timings={
            "ttft_seconds": round(ttft if ttft is not None else total, 6),
            "total_seconds": round(total, 6),
        },
        meta_info=meta_info if req.return_meta else None,
    )


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    runtime = require_engine()
    prompt = request_to_prompt(req)
    sampling_params = request_to_sampling_params(req)

    async def event_stream():
        start = time.perf_counter()
        ttft: float | None = None

        try:
            generator = await runtime.async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                stream=True,
            )
            async for chunk in generator:
                text = chunk.get("text", "")
                if text and ttft is None:
                    ttft = time.perf_counter() - start

                payload = {
                    "text": text,
                    "finish_reason": chunk.get("finish_reason"),
                    "meta_info": chunk.get("meta_info", {}),
                    "timings": {
                        "ttft_seconds": round(ttft, 6)
                        if ttft is not None
                        else None,
                        "elapsed_seconds": round(time.perf_counter() - start, 6),
                    },
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                if chunk.get("finish_reason") is not None:
                    break

            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Streaming generation failed")
            payload = {"error": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
