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
from transformers import AutoTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage
import torch

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
mistral_instruct_engine = None
tokenizer = None

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.put("/mistral_instruct_generate")
async def mistral_instruct_generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).

    request_dict:
    {
        "messages" [{'role':str,'content':str},....],
        "stream": bool,
        "sampling_params": **kwargs,
    }


    """
    request_dict = await request.json()
    messages = request_dict.pop("messages")     

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, continue_final_message=False)

    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)

    # import pdb;pdb.set_trace()
    # sampling_params.stop = tokenizer._eos_token  ###</s>
    sampling_params.stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

    request_id = random_uuid()

    results_generator = mistral_instruct_engine.generate(prompt,sampling_params,request_id)

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
            await mistral_instruct_engine.abort(request_id)
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
    mistral_instruct_engine_args = AsyncEngineArgs()
    mistral_instruct_engine_args.model="/dev/pretrained_models/mistralai/Mistral-7B-Instruct-v0.3"
    mistral_instruct_engine_args.tokenizer="/dev/pretrained_models/mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(mistral_instruct_engine_args.model)


    mistral_instruct_engine_args.tensor_parallel_size=4
    mistral_instruct_engine_args.config_format=ConfigFormat.HF
    mistral_instruct_engine_args.gpu_memory_utilization=0.6
    mistral_instruct_engine_args.dtype = torch.float32     ##########fp32不能正常运行，fp16效果较差
    mistral_instruct_engine = AsyncLLMEngine.from_engine_args(mistral_instruct_engine_args)


    uvicorn.run(app,
                host="0.0.0.0",
                port=5000,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)