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
parser.add_argument('--port', type=int, default=9881, help="服务器监听端口")
parser.add_argument('--version', type=str, default="v3", help="version")#尽量少修改代码，通过配置去影响代码行为

args = parser.parse_args()

client = None
cosyvoice = None
version = args.version#"v3"
special_token = "))))" if version=="v3" else "<|dream|>"
args.port = 9881 if version=="v3" else 9880
@asynccontextmanager
async def lifespan(app:FastAPI):
    global cosyvoice, client
    # init engine
    try:
        if version == "v3":
            client = OpenAI(
                # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
                api_key="EMPTY",
                base_url='http://127.0.0.1:8902/v1',#http://192.168.1.245:8901/
            )
        elif version == "v2":
            client = OpenAI(
                # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
                api_key="EMPTY",
                base_url='http://192.168.1.245:8901/',#http://192.168.1.245:8901/
            )
        else:
            Warning("Qwen-Omni version is not supported")
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
    print("inference text", text)
    #set_all_random_seed(random.randint(1,1e4))
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
            for _, j in enumerate(cosyvoice.inference_zero_shot(text, ref_text, prompt_speech_16k, zero_shot_spk_id="", stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                yield audio_stream
        else:#if isinstance(text,list) or isinstance(text,Generator):
            for _, j in enumerate(cosyvoice.inference_zero_shot(text, ref_text, prompt_speech_16k, zero_shot_spk_id="", stream=stream)):
                if openai:
                    yield j
                else:
                    output = j["tts_speech"]
                    audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
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
                if openai:
                    yield j
                else:
                    output = j["tts_speech"]
                    audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                    yield audio_stream  
    elif mode=="crosslingual":
        if isinstance(text,str):
            for _, j in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech_16k, stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                yield audio_stream
        else:#if isinstance(text,list) or isinstance(text,Generator):
            for _, j in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech_16k, stream=stream)):
                if openai:
                    yield j
                else:
                    output = j["tts_speech"]
                    audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                    yield audio_stream
    elif mode=="crosslingual-with-spk-id":
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
            for _, j in enumerate(cosyvoice.inference_cross_lingual(text, '', zero_shot_spk_id=spk_id, stream=stream)):
                output = j["tts_speech"]
                audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                yield audio_stream
        else:#if isinstance(text,list) or isinstance(text,Generator):
            for _, j in enumerate(cosyvoice.inference_cross_lingual(text, '', zero_shot_spk_id=spk_id, stream=stream)):
                if openai:
                    yield j
                else:
                    output = j["tts_speech"]
                    audio_stream = (output.numpy() * 32767).astype(np.int16).tobytes()
                    yield audio_stream
                    
    else:
        raise(f"{mode} mode is not supported")

def format_message(messages):
    if version=="v3":
        for message in messages:
            contents = message["content"]
            if isinstance(contents,list):
                for content in contents:
                    if content["type"] == "audio":
                        content["type"] = "input_audio"
                        content["input_audio"] = {}
                        if content["audio"].startswith("data:audio"):#如果带头，把头去掉
                            content["audio"] = content["audio"].split(',', 1)[1]
                        content["input_audio"]["data"] = content["audio"]
                        content["input_audio"]["format"] = "wav"
    return messages
def text_generator(messages, mode:str, lang="", is_cut=False, min_len=10):
    if mode.startswith("crosslingual") and lang:
        yield lang
    completion = client.chat.completions.create(
        model="/mnt/diskhd/Backup/DownloadModel/Qwen3-Omni-30B-A3B-Instruct/",
        messages=messages,
        stream=True,
        modalities=["text"],
        temperature=0.1
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

###################### 文本必须等到第一个音频chunk生成才会输出 ########################
def run_llm_stream_input0(messages, mode:str="zero-shot", ref_audio_path:str='reference.wav', ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', stream=True, modalities=["text"], openai=True, is_cut=False, min_len=10, lang=""):
    text = text_generator(messages, mode, lang, is_cut,min_len)    
    if mode=="zero-shot-with-spk-id" or mode=="crosslingual-with-spk-id":
        prompt_speech_16k = None
    else:
        prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
        
    start = 0
    for audio_stream in inference(mode, spk_id, text, ref_text, prompt_speech_16k, stream, openai):
        #text是一个generator类型，不能被序列化
        #当前text已经被消费掉了
        if openai:
            output_text, audio = audio_stream.get("text"), audio_stream.get("tts_speech")
            if output_text is not None:
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
    
from collections import deque

# =========================
# 方案2: 并行流处理，但是2次llm推理
# =========================
import queue, threading
def run_llm_stream_input_independent(messages, mode:str="zero-shot", ref_audio_path:str='reference.wav', 
                                 ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', 
                                 stream=True, modalities=["text"], openai=True, is_cut=False, min_len=10, lang=""):
    """
    并行的方案：文本和音频完全独立生成，但是2次llm推理
    """
    print("使用真正并行方案...")
    
    # 创建两个完全独立的文本生成器。相当于让llm做2次文本推理
    def create_text_gen():
        return text_generator(messages, mode, lang, is_cut, min_len)
    
    # 输出队列
    output_queue = queue.Queue(maxsize=200)
    
    # 文本处理线程
    def text_processor():
        try:
            print("文本处理线程启动...")
            text_count = 0
            for text_chunk in create_text_gen():
                print(f"文本线程生成块 {text_count}: {text_chunk[:50]}...")
                text_count += 1
                
                if openai:
                    chunk = ChatCompletionChunk(
                        id=f"",
                        created=int(time.time()),
                        model="",
                        choices=[Choice(
                            index=0,
                            delta=Delta(content=text_chunk),
                            finish_reason=None
                        )]
                    )
                    output_queue.put(('text', f"data: {json.dumps(chunk.dict())}\n\n"))
                else:
                    output_queue.put(('text', text_chunk))
            
            print(f"文本处理完成，共 {text_count} 块")
            output_queue.put(('text_done', None))
            
        except Exception as e:
            print(f"文本处理错误: {e}")
            output_queue.put(('text_error', str(e)))
    
    # 音频处理线程  
    def audio_processor():
        try:
            print("音频处理线程启动...")
            
            if mode=="zero-shot-with-spk-id" or mode=="crosslingual-with-spk-id":
                prompt_speech_16k = None
            else:
                prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
            
            audio_count = 0
            # 创建独立的文本生成器给inference
            for audio_result in inference(mode, spk_id, create_text_gen(), ref_text, prompt_speech_16k, stream, openai):
                print(f"音频线程生成块 {audio_count}")
                audio_count += 1
                
                if openai and audio_result.get("tts_speech") is not None:
                    audio = audio_result.get("tts_speech")
                    audio_stream_data = (audio.numpy() * 32767).astype(np.int16).tobytes()
                    
                    with io.BytesIO() as audio_io:
                        audio_io.write(audio_stream_data)
                        audio_io.seek(0)
                        audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                    
                    chunk = ChatCompletionChunk(
                        id=f"",
                        created=int(time.time()),
                        model="",
                        choices=[Choice(
                            index=0,
                            delta=Delta(audio={
                                "data": audio_base64,
                                "transcript": ""
                            }),
                            finish_reason=None
                        )]
                    )
                    output_queue.put(('audio', f"data: {json.dumps(chunk.dict())}\n\n"))
                elif not openai:
                    output_queue.put(('audio', audio_result))
            
            print(f"音频处理完成，共 {audio_count} 块")
            output_queue.put(('audio_done', None))
            
        except Exception as e:
            print(f"音频处理错误: {e}")
            import traceback
            traceback.print_exc()
            output_queue.put(('audio_error', str(e)))
    
    # 启动两个处理线程
    text_thread = threading.Thread(target=text_processor)
    audio_thread = threading.Thread(target=audio_processor)
    
    text_thread.daemon = True
    audio_thread.daemon = True
    
    text_thread.start()
    audio_thread.start()
    
    # 主流程：按顺序返回结果
    def output_stream():
        text_done = False
        audio_done = False
        
        while not (text_done and audio_done):
            try:
                msg_type, content = output_queue.get(timeout=1.0)
                
                if msg_type == 'text':
                    yield content
                elif msg_type == 'audio':
                    yield content
                elif msg_type == 'text_done':
                    text_done = True
                    print("文本流标记完成")
                elif msg_type == 'audio_done':
                    audio_done = True
                    print("音频流标记完成")
                elif msg_type in ['text_error', 'audio_error']:
                    print(f"处理错误: {content}")
                    break
                    
            except queue.Empty:
                print("输出队列超时等待...")
                continue
        
        if openai:
            yield "data: [DONE]\n\n"
        
        print("所有处理完成")
    
    return output_stream()

# =========================
# 方案1: 流式缓存分发器 (推荐)
# =========================
class StreamBuffer:
    """
    流式缓存器 - 核心思路：
    1. LLM生成文本时立即返回给客户端
    2. 同时缓存到buffer供后续音频处理使用
    3. 确保LLM只推理一次
    """
    
    def __init__(self):
        self.buffer = deque()
        self.finished = False
        self.error = None
        self.lock = threading.Lock()
        self.new_data_event = threading.Event()
    
    def add_chunk(self, chunk):
        """添加新的文本块"""
        with self.lock:
            self.buffer.append(chunk)
            self.new_data_event.set()
    
    def mark_finished(self):
        """标记生成完成"""
        with self.lock:
            self.finished = True
            self.new_data_event.set()
    
    def mark_error(self, error):
        """标记生成错误"""
        with self.lock:
            self.error = error
            self.finished = True
            self.new_data_event.set()
    
    def create_reader(self, name="reader"):
        """创建一个读取器，从buffer中读取数据"""
        def reader_gen():
            index = 0
            print(f"{name} 开始读取...")
            
            while True:
                with self.lock:
                    # 读取可用的新数据
                    while index < len(self.buffer):
                        chunk = self.buffer[index]
                        index += 1
                        print(f"{name} 读取块 {index-1}: {str(chunk)[:30]}...")
                        yield chunk
                    
                    # 检查是否完成
                    if self.error:
                        raise Exception(f"生成错误: {self.error}")
                    
                    if self.finished:
                        print(f"{name} 读取完成，共读取 {index} 块")
                        break
                
                # 等待新数据
                self.new_data_event.wait(timeout=5.0)
                self.new_data_event.clear()
        
        return reader_gen()

def run_llm_stream_input(messages, mode:str="zero-shot", ref_audio_path:str='reference.wav', 
                                       ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', 
                                       stream=True, modalities=["text"], openai=True, is_cut=False, min_len=10, lang=""):
    """
    方案1: 使用流式缓存器
    - LLM只推理一次
    - 文本立即流式返回
    - 同时缓存供音频处理
    """
    print("使用流式缓存器方案 (单次LLM推理)...")
    
    # 创建流式缓存器
    stream_buffer = StreamBuffer()
    
    # 音频结果队列
    audio_queue = queue.Queue(maxsize=100)
    
    # 音频处理线程
    def audio_processor():
        try:
            print("音频处理线程启动，等待文本数据...")
            
            # 准备TTS参数
            if mode=="zero-shot-with-spk-id" or mode=="crosslingual-with-spk-id":
                prompt_speech_16k = None
            else:
                prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
            
            # 从缓存器获取文本流
            inference_text_gen = stream_buffer.create_reader("inference")
            
            print("开始TTS推理...")
            audio_count = 0
            for audio_result in inference(mode, spk_id, inference_text_gen, ref_text, prompt_speech_16k, stream, openai):
                print(f"生成音频块 {audio_count}")
                audio_count += 1
                audio_queue.put(('audio', audio_result))
            
            print(f"音频处理完成，共生成 {audio_count} 块")
            audio_queue.put(('done', None))
            
        except Exception as e:
            print(f"音频处理错误: {e}")
            import traceback
            traceback.print_exc()
            audio_queue.put(('error', str(e)))
    
    # 启动音频处理线程
    audio_thread = threading.Thread(target=audio_processor)
    audio_thread.daemon = True
    audio_thread.start()
    
    # 主流程
    def combined_stream():
        try:
            print("开始LLM推理和文本流式返回...")
            text_count = 0
            
            # LLM推理一次，同时返回和缓存
            for text_chunk in text_generator(messages, mode, lang, is_cut, min_len):
                print(f"LLM生成文本块 {text_count}: {text_chunk[:50]}...")
                text_count += 1
                
                # 立即缓存供音频处理
                stream_buffer.add_chunk(text_chunk)
                
                # 立即返回给客户端
                if openai:
                    chunk = ChatCompletionChunk(
                        id=f"",
                        created=int(time.time()),
                        model="",
                        choices=[Choice(
                            index=0,
                            delta=Delta(content=text_chunk),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {json.dumps(chunk.dict())}\n\n"
                else:
                    yield text_chunk
            
            # 标记LLM生成完成
            stream_buffer.mark_finished()
            print(f"LLM推理完成，共生成 {text_count} 块文本")
            
        except Exception as e:
            print(f"LLM推理错误: {e}")
            stream_buffer.mark_error(str(e))
            return
        
        # 现在处理音频流
        print("LLM文本流结束，等待音频流...")
        audio_finished = False
        audio_count = 0
        
        while not audio_finished:
            try:
                msg_type, content = audio_queue.get(timeout=3.0)
                
                if msg_type == 'done':
                    audio_finished = True
                    print("音频流完成")
                elif msg_type == 'error':
                    print(f"音频错误: {content}")
                    audio_finished = True
                elif msg_type == 'audio':
                    audio_count += 1
                    print(f"返回音频块 {audio_count}")
                    
                    if openai and content.get("tts_speech") is not None:
                        audio = content.get("tts_speech")
                        audio_stream_data = (audio.numpy() * 32767).astype(np.int16).tobytes()
                        
                        with io.BytesIO() as audio_io:
                            audio_io.write(audio_stream_data)
                            audio_io.seek(0)
                            audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                        
                        chunk = ChatCompletionChunk(
                            id=f"",
                            created=int(time.time()),
                            model="",
                            choices=[Choice(
                                index=0,
                                delta=Delta(audio={
                                    "data": audio_base64,
                                    "transcript": ""
                                }),
                                finish_reason=None
                            )]
                        )
                        yield f"data: {json.dumps(chunk.dict())}\n\n"
                    elif not openai:
                        yield content
                        
            except queue.Empty:
                print("音频队列超时...")
                if not audio_thread.is_alive():
                    print("音频线程已结束")
                    break
        
        print(f"所有处理完成。文本: {text_count} 块, 音频: {audio_count} 块")
        
        if openai:
            yield "data: [DONE]\n\n"
    
    return combined_stream()

def run_llm(messages, mode:str="zero-shot", ref_audio_path:str='reference.wav', ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', stream=True, modalities=["text"], openai=True, is_cut=False, min_len=10, lang=""):
    completion = client.chat.completions.create(
        model="/mnt/diskhd/Backup/DownloadModel/Qwen3-Omni-30B-A3B-Instruct/",
        messages=messages,
        stream=True,
        modalities=["text"],
        temperature=0.1
    )
    if mode=="zero-shot-with-spk-id" or mode=="crosslingual-with-spk-id":
        prompt_speech_16k = None
    else:
        prompt_speech_16k = resample_wav_to_16khz(ref_audio_path)
    #以下做OpenAI兼容
    if mode.startswith("crosslingual"):
        text = lang
    else:
        text = ""
    start = 0
    punc = ",.?!，。？！"
    first = True
    pause = False
    index = 0
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
            if pause or chunk.choices[0].delta.content == special_token:
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
                print(text[start:])
                for audio_stream in inference(mode, spk_id, text[start:], ref_text, prompt_speech_16k, stream):
                    if openai:
                        index = index + 1
                        print("返回的音频片段",index)
                        #audio_np = np.frombuffer(audio_stream, dtype=np.int16)
                        #sf.write(f"tmp{index}.wav", audio_np, samplerate=24000)
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
                                    index=index,
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
            print(text[start:])
            for audio_stream in inference(mode, spk_id, text[start:], ref_text, prompt_speech_16k, stream):
                #走的都是这里，没走上面
                #这里tts是流式返回，一次只有一个音频片段，不是所有文本的音频片段，只返回了第一个音频片段后 yield "data: [DONE]\n\n"，导致后面没有返回给客户端
                if openai:
                    index = index + 1
                    print("最后返回的音频片段",index)
                    #audio_np = np.frombuffer(audio_stream, dtype=np.int16)
                    #sf.write(f"tmp{index}.wav", audio_np, samplerate=24000)
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
                                index=index,
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
            if openai:
                yield "data: [DONE]\n\n"
            
def run(text, mode:str="zero-shot", ref_audio_path:str='reference.wav', ref_text:str = '你能开那种，珍藏好多年都不褪色的发票吗', spk_id='female', stream_output=True, modalities=["text"], openai=True, is_cut=False, min_len=10, lang=""):
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
        lang = body.get("lang","")
        try:
            return StreamingResponse(
                    run(text, mode, ref_audio_path, ref_text, spk_id, stream_output, lang=lang),
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
        messages = format_message(messages)
        modalities = body.get("modalities",["audio"])
        
        mode = body.get("mode", "zero-shot-with-spk-id")
        ref_audio_path = body.get("ref_audio_path", "asset/zero_shot_prompt.wav")
        ref_text = body.get("ref_text", "希望你以后能够做的比我还好呦。")
        spk_id = body.get("spk_id", 'female')
        #if spk_id == "female":#女性，转成zero-shot，用这个参考音频
        #    mode = "zero-shot"
        #    ref_audio_path = "asset/zero_shot_prompt.wav"
        #    ref_text = "希望你以后能够做的比我还好呦。"
        openai = body.get("openai", False)
        is_cut = body.get("is_cut", False)
        min_len = body.get("min_len", 5)
        stream_input = body.get("stream_input", True) #写死，流式输入会导致数字发音有错
        stream_output = body.get("stream_output", True)
        lang = body.get("lang","")
        try:
            if stream_input:
                if openai:
                    return StreamingResponse(
                            run_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
                            media_type="text/event-stream"
                    )
                else:
                    return StreamingResponse(
                            run_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
                            media_type="application/octet-stream"
                    )
            else:
                if openai:
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
                        media_type="text/event-stream"
                    ) 
                else:       
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
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
        messages = format_message(messages)
        modalities = body.get("modalities",["text","audio"])
        
        mode = body.get("mode", "zero-shot-with-spk-id")
        ref_audio_path = body.get("ref_audio_path", "asset/zero_shot_prompt.wav")
        ref_text = body.get("ref_text", "希望你以后能够做的比我还好呦。")
        spk_id = body.get("spk_id", 'female')
        #if spk_id == "female":#女性，转成zero-shot，用这个参考音频
        #    mode = "zero-shot"
        #    ref_audio_path = "asset/zero_shot_prompt.wav"
        #    ref_text = "希望你以后能够做的比我还好呦。"            
        openai = body.get("openai", True)
        is_cut = body.get("is_cut", False)
        min_len = body.get("min_len", 5)
        stream_input = body.get("stream_input", True) #写死，流式输入会导致数字发音有错
        stream_output = body.get("stream_output", True)
        lang = body.get("lang","")
        try:
            if stream_input:
                if openai:
                    return StreamingResponse(
                            run_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
                            media_type="text/event-stream"
                    )
                else:
                    return StreamingResponse(
                            run_llm_stream_input(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
                            media_type="application/octet-stream"
                    )
            else:
                if openai:
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
                        media_type="text/event-stream"
                    ) 
                else:       
                    return StreamingResponse(
                        run_llm(messages, mode, ref_audio_path, ref_text, spk_id, stream_output, modalities, openai, is_cut, min_len, lang),
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
        
def fronted(text:str, is_generated:bool = False):
    normalized_text = cosyvoice.frontend.text_normalize(text)
    print(normalized_text)
    return normalized_text

@app.get("/fronted")
async def chat_completions(text:str, is_generated:bool = False):
    """兼容OpenAI格式的聊天补全API"""
    try:             
        # 准备提示
        try:
            return StreamingResponse(
                    fronted(text, is_generated),
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
