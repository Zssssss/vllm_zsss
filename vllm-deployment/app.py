import spaces
import torch
import gradio as gr
import tempfile
import os
import uuid
import scipy.io.wavfile
import time  
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed, AutoModel
import re



import transformers
import numpy as np
import librosa

model = AutoModel.from_pretrained('/dev/pretrained_models/fixie-ai/ultravox-v0_4', device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/dev/pretrained_models/fixie-ai/ultravox-v0_4")
pipe = transformers.pipeline(model=model,tokenizer=tokenizer, device="auto")

# path = "./user.wav"  # TODO: pass the audio here
# audio, sr = librosa.load(path, sr=16000)

turns = [
  {
    "role": "system",
    "content": "You are a friendly and helpful character. You love to answer questions for people."
  },
]
# pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=30)


@spaces.GPU
def transcribe(inputs):
    sample_rate, audio_data = inputs
    res = pipe({'audio': audio_data, 'turns': turns, 'sampling_rate': sample_rate}, max_new_tokens=30)

    return res
     

def clear():
    return ""

with gr.Blocks() as demo:
    with gr.Column():
        # gr.Markdown(f"# Realtime Whisper Large V3 Turbo: \n Transcribe Audio in Realtime. This Demo uses the Checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers.\n Note: The first token takes about 5 seconds. After that, it works flawlessly.")
        with gr.Row():
            with gr.Column():
                # input_audio_microphone = gr.Audio(streaming=True)
                input_audio_microphone = gr.Audio(
                    sources=["microphone"],
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        waveform_progress_color="#0066B4",
                        skip_length=2,
                        show_controls=False,
                    ),
                )
            with gr.Column():
                output = gr.Textbox(label="Transcription", value="")
        

        input_audio_microphone.stream(transcribe, [input_audio_microphone, output], [output], time_limit=45, stream_every=2, concurrency_limit=None)



demo.launch()