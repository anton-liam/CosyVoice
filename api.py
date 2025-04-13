import io
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice2Web
from cosyvoice.utils.file_utils import load_wav

import torch
import torchaudio

import uuid

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

def process_audio(tts_speeches, sample_rate=24000, format="wav"):
    """处理音频数据并返回响应"""
    buffer = io.BytesIO()
    audio_data = torch.concat(tts_speeches, dim=1)

    # 原始采样率（CosyVoice 默认为22050）
    original_sr = 24000
    
    # 如果目标采样率与原始采样率不同，进行重采样
    if sample_rate != original_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sample_rate)
        audio_data = resampler(audio_data)

    encoding = None
    if format == "opus":
        format = "ogg"
        encoding = 'opus'

    torchaudio.save(buffer, audio_data, sample_rate, encoding=encoding, format=format)
    buffer.seek(0)
    return buffer

@app.get("/ping")
async def create_zero_shot_spk():
   return {
       "status": 'active'
   }

@app.get("/tts")
async def inference_sft_by_spk(tts_text: str, spk_id: str, speed: float = 1.0, format:str = "wav"):
    model_output = lambda: cosyvoice.inference_sft_by_spk(tts_text, spk_id, speed=speed)
    
    def generate():
        for _, i in enumerate(model_output()):
            buffer = process_audio([i['tts_speech']], format=format)
            yield buffer.read()

    return StreamingResponse(generate(), media_type=f"audio/{format}", headers={
        "Content-Type": f"audio/{format}" ,
    })


@app.post("/speakers")
async def create_zero_shot_spk(name:str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    id = f"{uuid.uuid4()}_{name}" 
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    cosyvoice.create_zero_shot_spk(prompt_text, prompt_speech_16k, id)
    return {
        "id": id
    }

@app.delete("/speakers")
async def create_zero_shot_spk(name:str):
    cosyvoice.remove_spk(name)
    return 'ok'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--speaker_dir',
                        type=str,
                        default='speakers')
    args = parser.parse_args()

    try:
        cosyvoice = CosyVoice2Web(args.model_dir, args.speaker_dir)
    except Exception:
        raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)