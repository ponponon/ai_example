from loguru import logger
import time
from transformers import AutoTokenizer, AutoModel
import os
from typing import Dict, Tuple, Union, Optional
from torch.nn import Module
from transformers import AutoModel


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model


logger.debug(f'开始加载模型')
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b", trust_remote_code=True)
logger.debug(f'tokenizer down')
model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=1)
model = model.eval()
logger.debug(f'模型加载完毕')

tt = []

def func(filepath:str):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    lines = [
        content
    ]

    history=[]

    for line in lines:

        text = f"""{line}"""

        s = time.time()
        response, history = model.chat(tokenizer, text, history=history)
        e = time.time()

        tt.append(e-s)

        logger.debug(text)
        logger.debug(response)
        logger.debug(f'耗时: {round(e-s,3)} 秒')
        print('-----------------------------------------------')

        # with open('output-30.log', 'a', encoding='utf-8') as out_file:
        #     out_file.write(f'Question 👉 {text}')
        #     out_file.write('\n')
        #     out_file.write(f'Answer:👇\n{response}')
        #     out_file.write('\n\n-----------------------------------------------')
        #     out_file.write('\n')
        #     out_file.write('\n')
        #     out_file.write('\n')

    print(f'总耗时: {sum(tt)}秒')
    print(f'平均单个耗时: {sum(tt)/len(tt)}秒')


func('/home/pon/code/me/ai_example/examples/ChatGLM2_6B/q1.txt')