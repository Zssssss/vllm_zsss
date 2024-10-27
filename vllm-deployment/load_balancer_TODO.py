from flask import Flask, request, jsonify
import requests
import random

app = Flask(__name__)

endpoints = {
    'model_provider/model_1': 'http://0.0.0.0:8001',
    'model_provider/model_2': 'http://0.0.0.0:8002',
    'model_provider/model_3': 'http://0.0.0.0:8003',
    'model_provider/model_4': 'http://0.0.0.0:8004'
}

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model model_provider/model_1 --tensor-parallel-size 1 --chat-template ./chat.jinja --port 8001
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model model_provider/model_2 --tensor-parallel-size 1 --chat-template ./chat.jinja --port 8002
# CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model model_provider/model_3 --tensor-parallel-size 1 --chat-template ./chat.jinja --port 8003
# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model model_provider/model_4 --tensor-parallel-size 1 --chat-template ./chat.jinja --port 8004

def check_model(req, avail_endpoints, suffix_used, headers):
    data = req.get_json()
    model_name = data.get("model")
    if model_name:
        full_url = f"{avail_endpoints[model_name]}/v1/{suffix_used}"
        response = requests.post(full_url, json=data, headers=headers)
        return response.json(), response.status_code
    else:
        raise Exception("Model name not provided")
    
def merge_across_models(req, avail_endpoints, suffix_used, headers):
    final_response = None
    final_status_code = None
    for model_name, selected_endpoint in avail_endpoints.items():
        full_url = f"{selected_endpoint}/v1/{suffix_used}"
        response = requests.get(full_url, params=req.args, headers=headers)
        if final_response is None:
            final_response = response.json()
        else:
            final_response["data"].extend(response.json()["data"])
        final_status_code = final_status_code or response.status_code
    return final_response, final_status_code

@app.route('/v1/<path:suffix>', methods=['POST', 'GET'])
def api_load_balancer(suffix):

    headers = dict(request.headers)

    excluded_headers = ['Host', 'Content-Length', 'Content-Type']
    headers = {key: value for key, value in headers.items() if key not in excluded_headers}

    if request.method == 'POST':
        response_json, response_status_code = check_model(request, endpoints, suffix, headers)
    elif request.method == 'GET':
        response_json, response_status_code = merge_across_models(request, endpoints, suffix, headers)
    
    return jsonify(response_json), response_status_code

if __name__ == '__main__':
    app.run(debug=True, port=8000)