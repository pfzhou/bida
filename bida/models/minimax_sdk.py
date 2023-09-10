import requests
import json
import time
from enum import Enum

import requests
from bida import Util

class minimax():
    group_id=None
    api_key=None
    chat_completions_baseurl=None
    embeddings_baseurl=None
    
    POST_HEADERS = {"Content-Type": "application/json", "Authorization": ""}

    Default_Bot_Setting = [
                            {
                                "bot_name": "MM智能助理",
                                "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。"
                            }
                            ]
    Default_Reply_Constraints={ 
                                "sender_type": "BOT",
                                "sender_name": "MM智能助理"
                                }

    @classmethod
    def call(
        cls, 
        model: str,
        system_prompt: str,
        chat_messages: list,
        temperature: float,                # 默认0.9，范围 (0, 1.0]，不能为0
        max_tokens: int,
        stream: bool,  
        *args, **kwargs   
    ):
        url = minimax.chat_completions_baseurl

        if url is None:
            raise Exception("请指定有效的调用URL")
        
        messages = {}
        messages['model'] = model
        # 这里的system_prompt等同于openai的system prompt，CC生产力模式下可以不设定system prompt, CCP不支持system prompt
        if system_prompt is not None:
            messages["prompt"] = system_prompt
        # abab5 默认取值256，abab5.5 默认取值1024
        if max_tokens is not None:
            messages['tokens_to_generate'] = max_tokens
        # 设定流模式
        if stream is not None:
            messages["stream"] = stream
        # 设定temperature(0-1]
        if temperature is not None:
            if temperature == 0:        # 不能为0，将默认值调整为0.01
                temperature = 0.01       
            messages['temperature'] =temperature
        # 添加消息列表
        messages["messages"] = chat_messages

        # 对role_meta参数做特殊处理
        if 'role_meta' in kwargs:
            value = kwargs.pop('role_meta')
            for itemkey, itemvalue in value.items():
                messages[f'role_meta.{itemkey}'] = itemvalue

        # 添加其他参数
        for key, value in kwargs.items():
                messages[key] = value

        payload = json.dumps(messages)
        try:
            url = url + f'?GroupId={minimax.group_id}'
            headers = minimax.POST_HEADERS
            headers['Authorization'] = "Bearer " + minimax.api_key
            response = requests.request("POST", url, headers=headers, data=payload, stream=stream)
            if stream:
                if response.status_code == 200:
                    return response.iter_lines()
                else:
                    raise Exception(f"调用模型出错，状态码：{response.status_code}，错误描述：{response.text}")
            else:
                return response.json()
        except Exception as e:
            Util.log_error(e)
            raise e
        
    @classmethod
    def embeddings_call(
        cls, 
        input,
        model: str,
        embedding_type: str,
        ):
        '''
        输入文本以获取embeddings。说明：每个文本长度不超过 4096个token
        采用了query和db分离的算法方案，使用embedding_type指定生成类型，
        如果是要作为被检索文本: db，如果是作为检索文本: query。（不能混用）
        '''
        url = minimax.embeddings_baseurl
        if url is None:
            raise Exception("请指定有效的调用URL")
        
        messages = {}
        messages['model'] = model
        messages['type'] = embedding_type
        messages['texts'] = input
        
        payload = json.dumps(messages)
        try:
            url = url + f'?GroupId={minimax.group_id}'
            headers = minimax.POST_HEADERS
            headers['Authorization'] = "Bearer " + minimax.api_key
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.json()
        except Exception as e:
            Util.log_error(e)
            raise e
        
    @staticmethod
    def MessageObjWrapper(sender_name, text):
        message = {"sender_name": sender_name, "text": text}
        return message
    
    @staticmethod
    def MessageStrWrapper(sender_name, text):
        message = minimax.MessageObjWrapper(sender_name, text)
        return json.dumps(message, ensure_ascii=False)