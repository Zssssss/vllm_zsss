"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback

from client_wrappers import VLLMClient

parser = argparse.ArgumentParser()

# Server
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str, default=None)
parser.add_argument("--ssh_key_path", type=str, default=None)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))


def main():
    start_time = time.time()
    # Load api
    llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
        nonlocal llm

        while True:
            try:
                pred_list = llm.process_batch(prompts=input_list)
                break
            except Exception as e:
                traceback.print_exc()

        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list)

        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred['text'], str):
                pred_text = pred['text']
            elif len(pred['text']) > 0:
                pred_text = pred['text'][0]
            else:
                pred_text = ''

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }

    

    ####多线程调用llm
    
    threads = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]

        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)

            if (len(threads) == args.threads) or is_last_batch:
                for thread in threads:
                    thread.join()
                threads = []

                # dump the results in current processing window on disk
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + '\n')

                start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
