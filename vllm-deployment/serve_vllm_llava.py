import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
llava_dpo_engine = None

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)



@app.put("/llava_dpo_genetate")
async def llava_dpo_genetate(request:Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = llava_dpo_engine.generate(prompt,
                                        sampling_params,
                                        request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await llava_dpo_engine.abort(request_id)
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

    llava_dpo_engine_args = AsyncEngineArgs()
    llava_dpo_engine_args.model="/dev/pretrained_models/llava-hf/llava-interleave-qwen-7b-dpo-hf/"
    llava_dpo_engine_args.tensor_parallel_size=4
    llava_dpo_engine_args.config_format=ConfigFormat.HF
    llava_dpo_engine_args.gpu_memory_utilization=0.15
    llava_dpo_engine = AsyncLLMEngine.from_engine_args(llava_dpo_engine_args)

    uvicorn.run(app,
                host="0.0.0.0",
                port=5003,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)