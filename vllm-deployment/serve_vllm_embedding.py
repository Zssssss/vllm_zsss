import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.pooling_params import PoolingParams 
from vllm.utils import random_uuid


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
embedding_engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.put("/get_embedding")
async def get_embedding(request:Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    pooling_params = PoolingParams(**request_dict)
    request_id = random_uuid()

    results_generator = embedding_engine.encode(prompt,
                                        pooling_params,
                                        request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await embedding_engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--root-path",
    #     type=str,
    #     default=None,
    #     help="FastAPI root_path when app is behind a path based routing proxy")
    
    # app.root_path = args.root_path

    from vllm.config import ConfigFormat
    
    embedding_engine_args = AsyncEngineArgs()
    embedding_engine_args.model="/dev/pretrained_models/intfloat/e5-mistral-7b-instruct"
    embedding_engine_args.tensor_parallel_size=4
    embedding_engine_args.config_format=ConfigFormat.HF
    embedding_engine_args.gpu_memory_utilization=0.3
    embedding_engine_engine = AsyncLLMEngine.from_engine_args(embedding_engine_args)

    uvicorn.run(app,
                host="0.0.0.0",
                port=5001,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)