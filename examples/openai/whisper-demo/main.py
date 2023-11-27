import whisper
from whisper import Whisper
import time
import contextlib
import json


@contextlib.contextmanager
def timer(msg: str = None, logger=None):
    if not logger:
        from loguru import logger as loguru_logger
        logger = loguru_logger

    start = time.time()
    yield
    logger.debug(f'{msg}, used {round(time.time() - start,3)} s')


with timer('加载模型'):
    model: Whisper = whisper.load_model("medium")


with timer('转成文本耗时'):
    result = model.transcribe(
        "resource/audio/小说《知北游》凭什么被称为半部天书？.wav",
        temperature=(0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0),
        **{
            'verbose': True, 'task': 'transcribe', 'language': None,
            'best_of': 5, 'beam_size': 5, 'patience': None,
            'length_penalty': None, 'suppress_tokens': '-1',
            'initial_prompt': None, 'condition_on_previous_text': True,
            'fp16': True, 'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0, 'no_speech_threshold': 0.6,
            'word_timestamps': False,
            'prepend_punctuations': '"\'“¿([{-', 'append_punctuations': '"\'.。,，!！?？:：”)]}、'
        }
    )
    print(result)
    print(json.dumps(result, ensure_ascii=False))


# with timer('转成文本耗时'):
#     result = model.transcribe(
#         "resource/audio/小说《知北游》凭什么被称为半部天书？.wav"
#     )
#     print(result)
#     print(json.dumps(result, ensure_ascii=False))
