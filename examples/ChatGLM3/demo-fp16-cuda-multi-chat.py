from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])

import time

tt = []

from loguru import logger


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


func('/home/pon/code/me/ai_example/examples/ChatGLM3/q1.txt')