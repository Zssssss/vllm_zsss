GPUS="4" # GPU size for tensor_parallel.
MODEL_PATH="/home/ubuntu/zsss/vllm-deployment/fixie-ai/ultravox-v0_4"
python serve_vllm.py --model=${MODEL_PATH} --config-format hf --tensor-parallel-size=${GPUS} --dtype bfloat16 --disable-custom-all-reduce 




#####尽量别用bash这种，否则会有奇怪的bug，之前用这个model_path解析出的末尾突然多了个\r,导致不能正常从本地load