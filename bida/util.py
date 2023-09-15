import re
import sys
import os
import logging
from logging.handlers import TimedRotatingFileHandler

import tiktoken

from bida import Config

logger = logging.getLogger("bida")

__all__ = [
    "log_info",
    "log_debug",
    "log_warn",
    "log_error",
]

def _setlogger(logger):
    logger.setLevel(logging.DEBUG)  # 设定日志级别

    if not os.path.exists(Config.log_file_path):
        os.makedirs(Config.log_file_path)

    # 创建一个handler，每天切割一次日志文件
    file_handler = TimedRotatingFileHandler(Config.log_file_path + 'bida.log', when='D', interval=1)
    # 设定log文件记录记录级别
    file_handler.setLevel(Config.log_file_level)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)

_setlogger(logger)


def log_debug(message, **params):
    msg = logfmt(dict(message=message, **params))
    if Config.bida_debug:
        print(message, file=sys.stderr)
    logger.debug(msg)


def log_info(message, **params):
    msg = logfmt(dict(message=message, **params))
    if Config.bida_debug:
        print(message, file=sys.stderr)
    logger.info(msg)


def log_warn(message, **params):
    msg = logfmt(dict(message=message, **params))
    print(message, file=sys.stderr)
    logger.warn(msg)


def log_error(message, **params):
    msg = logfmt(dict(message=message, **params))
    print(message, file=sys.stderr)
    logger.error(msg)


def logfmt(props):
    logger.handlers[0].setLevel(Config.log_file_level)  # 运行中如果修改了log文件输出类型，动态修改
    def fmt(key, val):
        # Handle case where val is a bytes or bytesarray
        if hasattr(val, "decode"):
            val = val.decode("utf-8")
        # Check if val is already a string to avoid re-encoding into ascii.
        if not isinstance(val, str):
            val = str(val)
        if re.search(r"\s", val):
            val = repr(val)
        # key should already be a string
        if re.search(r"\s", key):
            key = repr(key)
        return "{key}={val}".format(key=key, val=val)

    return " ".join([fmt(key, val) for key, val in sorted(props.items())])

def get_text_token_count(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    en = tokenizer.encode(text)
    result = len(en)
    return result

def calculate_text_elements_count(text):
    # 计算英文单词数量
    words_count = len(re.findall(r'\b[a-zA-Z]+\b', text))
    # 计算标点符号数量，不包括汉字标点符号
    punctuation_count = len(re.findall(r'[^\w\s\u4e00-\u9fa5\uf900-\ufa2d]', text))
    # 计算汉字数量，包括汉字标点符号
    chinese_count = len(re.findall(r'[\u4e00-\u9fa5\uf900-\ufa2d]', text))
    # 计算其他非汉字字符数量
    other_characters_count = len(re.findall(r'[^\u4e00-\u9fa5\uf900-\ufa2d\w\s]', text)) - punctuation_count

    return words_count, punctuation_count, chinese_count, other_characters_count

def mask_key(key, show_len=3, mask_len=5):
    if len(key) <= 2 * show_len:
        return key
    else:
        return key[:show_len] + '*' * mask_len + key[-show_len:] 
