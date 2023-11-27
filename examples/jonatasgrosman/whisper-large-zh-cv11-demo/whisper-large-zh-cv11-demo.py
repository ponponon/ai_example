"""
https://huggingface.co/jonatasgrosman/whisper-large-zh-cv11
"""


from transformers import pipeline
import time
import contextlib


@contextlib.contextmanager
def timer(msg: str = None, logger=None):
    if not logger:
        from loguru import logger as loguru_logger
        logger = loguru_logger

    start = time.time()
    yield
    logger.debug(f'{msg}, used {round(time.time() - start,3)} s')


with timer('加载模型'):
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="jonatasgrosman/whisper-large-zh-cv11",
        device=0
    )

with timer('配置模型'):
    transcriber.model.config.forced_decoder_ids = (
        transcriber.tokenizer.get_decoder_prompt_ids(
            language="zh",
            task="transcribe"
        )
    )

with timer('翻译耗时'):
    transcription = transcriber("/home/pon/code/me/ai_example/26s.wav")
    print(transcription)
