import json
from typing import (Any, Tuple)

from bida import Util
from bida.core.model_api_base import ModelAPIBase
from bida.core.llm_api_base import LLMAPIBase
from bida.core.embedding_api_base import EmbeddingAPIBase
from bida.models.xfyun_sdk import xfyun_xinghuo

class xfyun_xinghuo_api(LLMAPIBase, EmbeddingAPIBase):
    '''
    讯飞星火大模型的API封装类，支持LLM Chat和Embeddings的调用。
    '''
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        xfyun_xinghuo.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])
        xfyun_xinghuo.secret_key = ModelAPIBase.get_config_value(self.api_config["secret_key"])
        xfyun_xinghuo.app_id = ModelAPIBase.get_config_value(self.api_config["app_id"])
        model_baseurls = self.api_config["model_baseurls"]
        if model_baseurls is not None:
            for key, value in model_baseurls.items():
                xfyun_xinghuo.model_baseurls[key] = ModelAPIBase.get_config_value(value)
        model_domains = self.api_config["model_domains"]
        if model_domains is not None:
            for key, value in model_domains.items():
                xfyun_xinghuo.model_domains[key] = ModelAPIBase.get_config_value(value)

        self.log_init(
            model_name=self.model_name, 
            app_id=Util.mask_key(xfyun_xinghuo.app_id),
            api_key=Util.mask_key(xfyun_xinghuo.api_key), 
            secret_key=Util.mask_key(xfyun_xinghuo.secret_key),
        )
    ################################################
    #  completions和chatcompletions的能力           #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return False
    
    def get_completion_content(self, response) -> Any:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_chatcompletion_content(self, response) -> str:
        if not response:
            raise Exception(f"生成出错。")
        result = ""
        for chunk in response:
            message, status = self.do_chunk(chunk)
            result += message
        return result

    def do_chunk(self, chunk):
        message = json.loads(chunk)
        if message['header']['code'] != 0:
            raise Exception(f"生成文本出错，代码：{message['header']['code']}，错误描述：{message['header']['message']}")
        result = message["payload"]["choices"]["text"][0]["content"]
        status =  message["payload"]["choices"]['status']
        return result, status
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        if not chunk:
            raise Exception(f"生成出错。")
        stop = False
        content, status = self.do_chunk(chunk[0])
        # 调用外部传递过来的处理函数处理回调返回的数据
        if content != '':
            content = self.process_stream_data(stream_callback, content, stream_callback_args)

        stop = True if status == 2 else False

        return content, stop, None
    
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        i = j = 0
        if stream:
            message = response[0]
        else:
            message = response[-1]
        message = json.loads(message)
        if message["payload"]["choices"]['status'] == 2:
            i = message["payload"]["usage"]["text"]["prompt_tokens"]
            j = message["payload"]["usage"]["text"]["total_tokens"]
        
        return i, j
    
    def get_text_token_count(self, text) -> int:
        # https://www.xfyun.cn/doc/spark/Web.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E
        # token统计信息，tokens 约等于1.5个中文汉字 或者 0.8个英文单词
        words_count, punctuation_count, chinese_count, other_characters_count = Util.calculate_text_elements_count(text)
        # 忽略标点符号，非汉字的其他语言字符按2个计算
        sum = chinese_count*1.5 + words_count*0.8 + other_characters_count*2
        return sum
    
    def textcompletion(
            self, 
            prompt, 
            temperature, 
            max_tokens, 
            stream, 
            *args, **kwargs
            ):
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def chatcompletion(
            self,
            prompt,
            chat_messages,
            temperature,
            max_tokens,
            stream,
            *args, **kwargs
            ):
        chat_messages.append({"role": 'user', "content": prompt})
        if temperature == 0:
            temperature = 0.01
        response = xfyun_xinghuo.call(
            model=self.model_name,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            *args, **kwargs
            )
        return response
    
    #######################################################
    #  embedding 向量生成的能力                            #
    #  实现 EmbeddingAPIBase 中1个@abstractmethod 方法    #
    #######################################################

    def embeddingcompletion(
            self, 
            inputText,
            *args, **kwargs
            ):
        # 不支持列表的批处理，只能一个一个调用
        result = {}
        for i, prompt in enumerate(inputText):
            response = xfyun_xinghuo.embeddings_call(
                model=self.model_name,
                text=prompt, 
                *args, **kwargs
                )
            em = json.loads(response.content.decode('utf-8'))
            if em['header']['code'] != 0:
                raise Exception(f"调用xfyun生成embedding出错，错误码：{em['header']['code']}，错误描述：{em['header']['message']}")
            
            item={}
            item['index'] = i
            item['embedding'] = json.loads(em["payload"]["text"]["vector"])
            if 'data' not in result:
                result['data'] = []
            result['data'].append(item)

            if 'usage' not in result:
                result['usage'] = {}
                result['usage']["prompt_tokens"] = 0
                result['usage']["total_tokens"] = 0

        return result
    