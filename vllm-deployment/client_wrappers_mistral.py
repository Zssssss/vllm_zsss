import abc
import json
import multiprocessing
import os
import re
import sys
import time
import requests
import traceback
from pathlib import Path
from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage


class Client(abc.ABC):
    def __init__(
        self,
        server_host,
        server_port='5000',
        ssh_server=None,
        ssh_key_path=None,
        **generation_kwargs
    ):
        self.server_host = server_host
        self.server_port = server_port
        # self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        # self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        self.ssh_server, self.ssh_key_path = ssh_server, ssh_key_path
        self.generation_kwargs = generation_kwargs
        
    @abc.abstractmethod
    def _single_call(
        self,
        prompts,
    ):
        pass

    def __call__(
        self,
        prompt: str,
        **kwargs
    ):
        request = self.generation_kwargs
        # prompts are added later
        request['prompts'] = [f'{prompt}']
        if 'others' in kwargs:
            request['others'] = kwargs['others']

        outputs = self._single_call(**request)
        response = {'text': outputs}
        return response
        
    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request, route="generate"):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            outputs = sshtunnel_request.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        else:
            outputs = requests.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        return outputs

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        num_threads = max(96, multiprocessing.cpu_count() * 16)
        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for prompt in prompts:
                futures.append(
                    executor.submit(
                        self.__call__,
                        prompt,
                        **kwargs,
                    )
                )
            rets = [f.result() for f in futures]
        return rets


class VLLMClient(Client):
    def _single_call(
        self,
        messages,
        *args, **kwargs
        # max_tokens=1024,
        # temperature=0.0,
        # top_p=1.0,
        # top_k=10,
        # random_seed,
        # stop: List[str],
    ):
        request = {
            "messages": [{'role':message.role,'content':message.content} for message in messages],
            # "max_tokens": tokens_to_generate_max,
            # "temperature": temperature,
            # "top_k": top_k,
            # "top_p": top_p,
            # "stop": stop,
        }
        request.update(kwargs)
        # TODO: random seed is not supported?
        outputs = self._send_request(request, route="mistral_instruct_generate")
        outputs = outputs['text'][0]
        return outputs


if __name__ == "__main__":
    llm = VLLMClient(server_host="127.0.0.1", server_port="5000")
    outputs = llm._single_call(messages=[])