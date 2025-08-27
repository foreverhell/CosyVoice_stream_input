#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import librosa
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
from io import BytesIO
import base64,io
from openai import OpenAI
import torch
import random

logger = init_logger('vllm.cosyvoice')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="/mnt/disk2/home/yiyangzhe/CosyVoice_stream_input/pretrained_models/CosyVoice2-0.5B/")#"../CosyVoice_stream_input/pretrained_models/CosyVoice2-0.5B"
#server
parser.add_argument('--host', type=str, default='0.0.0.0', help="服务器监听地址")
parser.add_argument('--port', type=int, default=9880, help="服务器监听端口")

args = parser.parse_args()

client = None
cosyvoice = None
@asynccontextmanager
async def lifespan(app:FastAPI):
    global cosyvoice, client
    # init engine
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="EMPTY",
            base_url='http://192.168.1.245:8901/',
        )
    except:
        Warning("Qwen-Omni server has not been started")
    cosyvoice = CosyVoice2(args.model, load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    warmup(2)
    yield

def warmup(times=2):
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for i in tqdm(range(times)):
        set_all_random_seed(i)
        for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=True)):
            continue
            
app = FastAPI(
    title="cosyvoice Compatible API", 
    description="API compatible with OpenAI format for cosyvoice model",
    lifespan=lifespan
)

def pcm_2_waveform(pcm_data: bytes) -> np.array:
    if len(pcm_data) & 1:
        pcm_data = pcm_data[:-1] # 保证偶数个字节
    int16_array = np.frombuffer(pcm_data, dtype=np.int16)
    waveform = int16_array.astype(np.float32) / (1<<15)
    return waveform

def resample_wav_to_16khz(input_filepath: str):
    if input_filepath.startswith("data:audio"):
        audio_bytes = base64.b64decode(input_filepath.split(',', 1)[1])
        with io.BytesIO(audio_bytes) as f:
            data = load_wav(f,16000)          
    elif os.path.exists(input_filepath):
        data = load_wav(input_filepath,16000)
    else:
        data = None
    return data

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
    
    def dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

class Delta:
    def __init__(self, role=None, content=None, audio=None):
        self.role = role
        self.content = content
        self.audio = audio
    
    def dict(self):
        result = {}
        if self.role is not None:
            result["role"] = self.role
        if self.content is not None:
            result["content"] = self.content
        if self.audio is not None:
            result["audio"] = self.audio
        return result

class Choice:
    def __init__(self, index, delta, finish_reason=None):
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason
    
    def dict(self):
        return {
            "index": self.index,
            "delta": self.delta.dict(),
            "finish_reason": self.finish_reason
        }

class ChatCompletionChunk:
    def __init__(self, id, created, model, choices, usage=None):
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    
    def dict(self):
        result = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.dict() for choice in self.choices]
        }
        if self.usage is not None:
            result["usage"] = self.usage.dict()
        return result

async def process_output_queue(output_queue, response_queue, include_audio=True, request_id="", model_name=""):
    """处理输出队列，将结果转换为OpenAI兼容格式"""
    try:
        text_content = ""
        for output in output_queue:
            if output is None:
                usage = Usage(
                )            
                await response_queue.put({
                    "type": "usage",
                    "data": usage.dict()
                })
                # 结束流
                await response_queue.put(None)
                break
            
            if hasattr(output, 'text') and output.text:
                # 处理文本输出
                if output.text:
                    new_text = output.text
                    text_content = text_content + output.text
                    
                    if new_text:                        
                        chunk = ChatCompletionChunk(
                            id=f"chatcmpl-{request_id}",
                            created=int(time.time()),
                            model=model_name,
                            choices=[
                                Choice(
                                    index=0,
                                    delta=Delta(
                                        content=new_text
                                    ),
                                    finish_reason=None
                                )
                            ]
                        )
                        
                        await response_queue.put({
                            "type": "text",
                            "data": chunk.dict()
                        })
            
            elif hasattr(output, 'audio') and output.audio and include_audio:
                # 处理音频输出
                if isinstance(output.audio, torch.Tensor) or isinstance(output.audio, np.ndarray):#数组
                    audio_stream = (output.audio.numpy() * 32767).astype(np.int16).tobytes()
                else:#流
                    audio_stream = output.audio
                # 将音频数据编码为base64
                with io.BytesIO() as audio_io:
                    #sf.write(audio_io, audio_data, 24000, format='RAW', subtype="PCM_16")#不能以WAV打包，包含头信息，只能分段播放
                    audio_io.write(audio_stream)
                    audio_io.seek(0)
                    audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                
                # 创建音频响应块
                chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{request_id}",
                    created=int(time.time()),
                    model=model_name,
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(
                                audio={
                                    "data": audio_base64,
                                    "transcript": text_content
                                }
                            ),
                            finish_reason=None
                        )
                    ]
                )
                
                await response_queue.put({
                    "type": "audio",
                    "data": chunk.dict()
                })
    
    except Exception as e:
        logger.exception(f"Error processing output queue: {e}")
        # 发送错误到响应队列
        await response_queue.put({
            "type": "error",
            "data": str(e)
        })
        # 结束流
        await response_queue.put(None)

async def stream_response(queue):
    """流式响应生成器"""
    try:
        while True:
            item = await queue.get()
            if item is None:
                # 流结束
                yield "data: [DONE]\n\n"
                break
            
            if item["type"] == "error":
                # 错误信息
                yield f"data: {json.dumps({'error': item['data']})}\n\n"
                break
            
            # 正常数据
            yield f"data: {json.dumps(item['data'])}\n\n"
    
    except Exception as e:
        logger.exception(f"Error in stream response: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


def inference(mode, spk_id, text, ref_text, prompt_speech_16k, stream=True, openai=False, modalities = ["audio"]):
    set_all_random_seed(random.randint(1,1e4))
    #这个在阻塞获得文本
    if mode=="sft":
        if spk_id=="":
            spk_id = "中文女"
        elif isinstance(spk_id, str):
            if spk_id not in cosyvoice.list_available_spks():
                spk_id = "中文女"
        elif isinstance(spk_id,torch.Tensor):#[1,192]，最好不要这样传；或者修改female的这两个参数？
            cosyvoice.frontend.spk2info["tmp"] = {}
            cosyvoice.frontend.spk2info["tmp"]["embedding"] = spk_id
            spk_id = "tmp"
        else:
            raise(f"{type(spk_id)} format spk_id is not supported")
        if cosyvoice.frontend.spk2info[spk_id].get("embedding") is None:
            if cosyvoice.frontend.spk2info[spk_id].get("flow_embedding") is not None:
                cosyvoice.frontend.spk2info[spk_id]["embedding"] = cosyvoice.frontend.spk2info[spk_id]["flow_embedding"]
            elif cosyvoice.frontend.spk2info[spk_id].get("llm_embedding") is not None:
                cosyvoice.frontend.spk2info[spk_id]["embedding"] = cosyvoice.frontend.spk2info[spk_id]["llm_embedding"]
            else:
                raise("Embedding is NEEDED!")
        for _, j in enumerate(cosyvoice.inference_sft(text, spk_id=spk_id, stream=stream)):
            output = j["tts_speech"]
            audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
            yield audio_stream    
    elif mode=="zero-shot":
        if isinstance(text,str):
            for _, j in enumerate(cosyvoice.inference_zero_shot(text, ref_text, prompt_speech_16k, stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                yield audio_stream
        else:#if isinstance(text,list) or isinstance(text,Generator):
            for _, j in enumerate(cosyvoice.inference_zero_shot(text, ref_text, prompt_speech_16k, stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                if openai:
                    yield j
                else:
                    yield audio_stream
    elif mode=="zero-shot-with-spk-id":
        print(cosyvoice.list_available_spks())
        if spk_id=="":
            spk_id = "female"
        elif isinstance(spk_id, str):
            if spk_id not in cosyvoice.list_available_spks():
                spk_id = "female"
        elif isinstance(spk_id,torch.Tensor):#[1,192]，最好不要这样传；或者修改female的这两个参数？
            cosyvoice.frontend.spk2info["tmp"] = {}
            cosyvoice.frontend.spk2info["tmp"]["llm_embedding"] = spk_id
            cosyvoice.frontend.spk2info["tmp"]["flow_embedding"] = spk_id
            spk_id = "tmp"
        else:
            raise(f"{type(spk_id)} format spk_id is not supported")
        if cosyvoice.frontend.spk2info[spk_id].get("flow_embedding") is None and cosyvoice.frontend.spk2info[spk_id].get("llm_embedding") is None:
            if cosyvoice.frontend.spk2info[spk_id].get("embedding") is not None:
                cosyvoice.frontend.spk2info[spk_id]["flow_embedding"] = cosyvoice.frontend.spk2info[spk_id]["embedding"]
                cosyvoice.frontend.spk2info[spk_id]["llm_embedding"] = cosyvoice.frontend.spk2info[spk_id]["embedding"]
            else:
                raise("Embedding is NEEDED!")
        if isinstance(text,str):
            for _, j in enumerate(cosyvoice.inference_zero_shot(text, '', '', zero_shot_spk_id=spk_id, stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                yield audio_stream
        else:#if isinstance(text,list) or isinstance(text,Generator):
            for _, j in enumerate(cosyvoice.inference_zero_shot(text, '', '', zero_shot_spk_id=spk_id, stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                if openai:
                    yield j
                else:
                    yield audio_stream
    elif mode=="crosslingual":
        for _, j in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech_16k, stream=stream)):
            output = j["tts_speech"]
            audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
            yield audio_stream
    else:
        raise(f"{mode} mode is not supported")


def text_generator(messages, is_cut=False, min_len=10):
    completion = client.chat.completions.create(
        model="qwen-omni",
        messages=messages,
        stream=True,
        modalities=["text"],
        extra_body={
            "is_first":False,
            "request_id":None,
            }
    )
    #以下做OpenAI兼容
    text = ""
    start = 0
    punc = ",.?!，。？！"
    first = True
    for chunk in completion:
        if first:
            min_lens = 1
        else:
            min_lens = min_len
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
            #text and text[-1] in punc and
            if (((not is_cut) or (len(text)-start >= min_lens))):#10个中文字符/在标点处做推理；英文不能按照字符，应该按照
                if first:
                    first = False
                    #first_audio_path = "/disk2/home/yiyangzhe/GPT-SoVITS_v2/reference0.wav"
                    #first_audio = resample_wav_to_16khz(first_audio_path)
                    #for audio_stream in inference(mode, spk_id, text, start, "你能开那种", first_audio):
                    #    yield audio_stream
                print(text[start:])
                yield text[start:]
                start = len(text)
        elif start != len(text):#推理最后一段可能长度不够的
            print(text[start:])
            yield text[start:]
            start = len(text)
            
            
def rum_llm_stream_input(messages, mode:str="zero-shot", ref_audio_path:str='reference.wav', ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', stream=True, modalities=["text"], openai=True, is_cut=False, min_len=10):
    text = text_generator(messages,is_cut,min_len)
    if mode=="zero-shot-with-spk-id":
        prompt_speech_16k = None
    else:
        prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
        
    start = 0
    for audio_stream in inference(mode, spk_id, text, ref_text, prompt_speech_16k, stream, openai):
        #text是一个generator类型，不能被序列化
        #当前text已经被消费掉了
        if openai:
            output_text, audio = audio_stream.get("text"), audio_stream.get("tts_speech")
            output_text = "".join(output_text)
            if output_text[start:]:
                chunk = ChatCompletionChunk(
                    id=f"",
                    created=int(time.time()),
                    model="",
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(
                                content=output_text[start:]
                            ),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                start = len(output_text)
            if audio is not None:
                audio_stream = (audio.numpy() * 32767).astype(np.int16).tobytes()
                #音频
                with io.BytesIO() as audio_io:
                    #sf.write(audio_io, audio_data, 24000, format='RAW', subtype="PCM_16")#不能以WAV打包，包含头信息，只能分段播放
                    audio_io.write(audio_stream)
                    audio_io.seek(0)
                    audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                
                # 创建音频响应块
                chunk = ChatCompletionChunk(
                    id=f"",
                    created=int(time.time()),
                    model="",
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(
                                audio={
                                    "data": audio_base64,
                                    "transcript": ""
                                }
                            ),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
        else:
            yield audio_stream
    if openai:
        yield "data: [DONE]\n\n"
        
def run_llm(messages, mode:str="zero-shot", ref_audio_path:str='reference.wav', ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', stream=True, modalities=["text"], openai=True, is_cut=False, min_len=10):
    completion = client.chat.completions.create(
        model="qwen-omni",
        messages=messages,
        stream=True,
        modalities=["text"],
        extra_body={
            "is_first":False,
            "request_id":None,
            }
    )
    prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
    #以下做OpenAI兼容
    text = ""
    start = 0
    punc = ",.?!，。？！"
    first = True
    pause = False
    for chunk in completion:
        if first:
            min_lens = 5
        else:
            min_lens = min_len
        if chunk.choices and chunk.choices[0].delta.content:
            if openai:
                chunk = ChatCompletionChunk(
                    id=f"",
                    created=int(time.time()),
                    model="",
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(
                                content=chunk.choices[0].delta.content
                            ),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
            if pause or chunk.choices[0].delta.content == "<|dream|>":
                pause = True
            else:
                text += chunk.choices[0].delta.content
            #text and text[-1] in punc and 
            if (text and text[-1] in punc and ((not is_cut) or (len(text)-start >= min_lens))):#10个中文字符/在标点处做推理；英文不能按照字符，应该按照
                if first:
                    first = False
                    #first_audio_path = "/disk2/home/yiyangzhe/GPT-SoVITS_v2/reference0.wav"
                    #first_audio = resample_wav_to_16khz(first_audio_path)
                    #for audio_stream in inference(mode, spk_id, text, start, "你能开那种", first_audio):
                    #    yield audio_stream
                #else:
                for audio_stream in inference(mode, spk_id, text[start:], ref_text, prompt_speech_16k, stream):
                    if openai:
                        with io.BytesIO() as audio_io:
                            #sf.write(audio_io, audio_data, 24000, format='RAW', subtype="PCM_16")#不能以WAV打包，包含头信息，只能分段播放
                            audio_io.write(audio_stream)
                            audio_io.seek(0)
                            audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                        
                        # 创建音频响应块
                        chunk = ChatCompletionChunk(
                            id=f"",
                            created=int(time.time()),
                            model="",
                            choices=[
                                Choice(
                                    index=0,
                                    delta=Delta(
                                        audio={
                                            "data": audio_base64,
                                            "transcript": ""
                                        }
                                    ),
                                    finish_reason=None
                                )
                            ]
                        )
                        yield f"data: {json.dumps(chunk.dict())}\n\n"
                    else:
                        yield audio_stream
                start = len(text)
        elif start != len(text):#推理最后一段可能长度不够的
            for audio_stream in inference(mode, spk_id, text[start:], ref_text, prompt_speech_16k, stream):
                if openai:
                    with io.BytesIO() as audio_io:
                        #sf.write(audio_io, audio_data, 24000, format='RAW', subtype="PCM_16")#不能以WAV打包，包含头信息，只能分段播放
                        audio_io.write(audio_stream)
                        audio_io.seek(0)
                        audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                    
                    # 创建音频响应块
                    chunk = ChatCompletionChunk(
                        id=f"",
                        created=int(time.time()),
                        model="",
                        choices=[
                            Choice(
                                index=0,
                                delta=Delta(
                                    audio={
                                        "data": audio_base64,
                                        "transcript": ""
                                    }
                                ),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {json.dumps(chunk.dict())}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield audio_stream
            start = len(text)
            
def run(text, mode:str="zero-shot", ref_audio_path:str='reference.wav', ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', stream_output=True, modalities=["text"], openai=True, is_cut=False, min_len=10):
    prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
    for audio_stream in inference(mode, spk_id, text, ref_text, prompt_speech_16k, stream_output):
        yield audio_stream

@app.post("/tts")
async def chat_completions(request: Request):
    """兼容OpenAI格式的聊天补全API"""
    try:
        # 解析请求体
        body = await request.json()
        # 获取必需字段
        text = body.get("text",None)
        
        mode = body.get("mode", "zero-shot-with-spk-id")
        ref_audio_path = body.get("ref_audio_path", "reference.wav")
        ref_text = body.get("ref_text", "你能开那种，珍藏好多年都不褪色的发票吗")
        spk_id = body.get("spk_id", 'female')
        stream_output = body.get("stream_output", True)
        try:
            return StreamingResponse(
                    run(text, mode, ref_audio_path, ref_text, spk_id, stream_output),
                    media_type="application/octet-stream"
            )   
        except Exception as e:
            logger.exception('Error {e} in run_cosyvoice_engine')
        
    except Exception as e:
        logger.exception(f"解析请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))
                
@app.post("/llm")
async def chat_completions(request: Request):
    """兼容OpenAI格式的聊天补全API"""
    try:
        # 解析请求体
        body = await request.json()
        # 获取必需字段
        messages = body.get("messages",None)
        modalities = body.get("modalities",["audio"])
        
        mode = body.get("mode", "zero-shot-with-spk-id")
        ref_audio_path = body.get("ref_audio_path", "reference.wav")
        ref_text = body.get("ref_text", "你能开那种，珍藏好多年都不褪色的发票吗")
        spk_id = body.get("spk_id", 'female')
        openai = body.get("openai", False)
        is_cut = body.get("is_cut", False)
        min_len = body.get("min_len", 5)
        stream_input = body.get("stream_input", True)
        stream_output = body.get("stream_output", True)
        '''
        if mode=="zero-shot-with-spk-id":
            mode="zero-shot"
            if spk_id=="male":
                ref_audio_path = "male_omni.wav"
                ref_text = "对啊，猫头鹰特别，他的眼睛又大又圆，晚上还能看清楚东西呢"
            elif spk_id=="female":
                ref_audio_path = "female_omni.wav"
                ref_text = "对啊，猫头鹰特别，他的眼睛又大又圆，晚上还能看清楚东西呢"
            elif spk_id=="male_moss":
                ref_audio_path = "male_moss.wav"
                ref_text = "如果大家想听到更丰富更及时的直播内容，记得在周一到周五准时进入直播间和大家一起"
        '''
        try:
            if stream_input:
                if openai:
                    return StreamingResponse(
                            rum_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                            media_type="text/event-stream"
                    )
                else:
                    return StreamingResponse(
                            rum_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                            media_type="application/octet-stream"
                    )
            else:
                if openai:
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                        media_type="text/event-stream"
                    ) 
                else:       
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                        media_type="application/octet-stream"
                    )
                
        except Exception as e:
            logger.exception('Error {e} in run_cosyvoice_engine')
        
    except Exception as e:
        logger.exception(f"解析请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/completions")
async def chat_completions(request: Request):
    """兼容OpenAI格式的聊天补全API"""
    try:
        # 解析请求体
        body = await request.json()
        # 获取必需字段
        messages = body.get("messages",None)
        modalities = body.get("modalities",["text","audio"])
        
        mode = body.get("mode", "zero-shot-with-spk-id")
        ref_audio_path = body.get("ref_audio_path", "reference.wav")
        ref_text = body.get("ref_text", "你能开那种，珍藏好多年都不褪色的发票吗")
        spk_id = body.get("spk_id", 'female')
        openai = body.get("openai", True)
        is_cut = body.get("is_cut", False)
        min_len = body.get("min_len", 5)
        stream_input = body.get("stream_input", True)
        stream_output = body.get("stream_output", True)
        '''
        if mode=="zero-shot-with-spk-id":
            mode="zero-shot"
            if spk_id=="male":
                ref_audio_path = "male_omni.wav"
                ref_text = "对啊，猫头鹰特别，他的眼睛又大又圆，晚上还能看清楚东西呢"
            elif spk_id=="female":
                ref_audio_path = "female_omni.wav"
                ref_text = "对啊，猫头鹰特别，他的眼睛又大又圆，晚上还能看清楚东西呢"
            elif spk_id=="male_moss":
                ref_audio_path = "male_moss.wav"
                ref_text = "如果大家想听到更丰富更及时的直播内容，记得在周一到周五准时进入直播间和大家一起"
        '''
        try:
            if stream_input:
                if openai:
                    return StreamingResponse(
                            rum_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                            media_type="text/event-stream"
                    )
                else:
                    return StreamingResponse(
                            rum_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                            media_type="application/octet-stream"
                    )
            else:
                if openai:
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                        media_type="text/event-stream"
                    ) 
                else:       
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len),
                        media_type="application/octet-stream"
                    )
        except Exception as e:
            logger.exception('Error {e} in run_cosyvoice_engine')
        
    except Exception as e:
        logger.exception(f"解析请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/llm")
async def chat_completions(messages:list=None, modalities:list=["text"], mode:str = "zero-shot-with-spk-id", ref_audio_path:str="", ref_text:str="", spk_id='female', openai=False, is_cut=False, min_len=5):
    """兼容OpenAI格式的聊天补全API"""
    try:             
        # 准备提示
        try:
            return StreamingResponse(
                    run_llm(messages, mode, ref_audio_path, ref_text, spk_id, modalities, openai, is_cut, min_len),
                    media_type="application/octet-stream"
            )   
        except Exception as e:
            logger.exception('Error {e} in run_cosyvoice_engine')
    except:
        logger.exception('Error {e} in run_cosyvoice_engine')

def main():
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
