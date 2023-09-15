from http import HTTPStatus
from typing import (Any, Tuple)
import json

from bida.core.model_api_base import ModelAPIBase
from bida.core.llm_api_base import LLMAPIBase
from bida.core.embedding_api_base import EmbeddingAPIBase
from bida import Util

import zhipuai

class zhipuai_api(LLMAPIBase, EmbeddingAPIBase):
    
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        zhipuai.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])

        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(zhipuai.api_key), 
        )

    ################################################
    #  completions和chatcompletions的能力          #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return False
    
    def get_completion_content(self, response) -> str:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_completion_stream_content(self, response, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_chatcompletion_content(self, response) -> Any:
        if not response["success"]:
            raise Exception(f"调用chatglm模型出错，错误码：{response['code']}，错误信息：{response['msg']}")
        result = response["data"]['choices'][0]['content'].strip()
        return result
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        # event: add 增量，finish 结束，error 错误，interrupted 中断
        finish_reason = chunk.event
        stop = True if finish_reason != "add" else False
        
        content = chunk.data

        if finish_reason == "error" or finish_reason == "interrupted":
            raise Exception(f"调用chatglm模型出错，错误信息：{content}")
        
        content = self.process_stream_data(stream_callback, content, stream_callback_args)

        return content, stop, None
    
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        if stream:
            j = json.loads(response.meta)["usage"]["total_tokens"]
        else:
            j = response["data"]["usage"]["total_tokens"]
        return 0, j
    
    def get_text_token_count(self, text) -> int:
        return 0
    
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
        if stream:
            response = zhipuai.model_api.sse_invoke(
                model=self.model_name,
                prompt=chat_messages,
                temperature=temperature,
                *args, **kwargs
                )
        else:
            response = zhipuai.model_api.invoke(
                model=self.model_name,
                prompt=chat_messages,
                *args, **kwargs
                )
        return response
        
    #######################################################
    #  embedding 向量生成的能力                            #
    #  实现 EmbeddingAPIBase 中1个@abstractmethod 方法     #
    #######################################################

    def embeddingcompletion(self, 
                            inputText,
                            *args, **kwargs
                            ):
        # 不支持列表的批处理，只能一个一个调用
        result = {}
        for i, prompt in enumerate(inputText):
            em = zhipuai.model_api.invoke(
                model=self.model_name,
                prompt=prompt, 
                )
            
            if not em["success"]:
                raise Exception(f"调用chatglm生成embedding出错，错误码：{em['code']}，错误信息：{em['msg']}")
            
            item={}
            item['index'] = i
            item['embedding'] = em['data']['embedding']
            if 'data' not in result:
                result['data'] = []
            result['data'].append(item)

            if 'usage' not in result:
                result['usage'] = {}
                result['usage']["prompt_tokens"] = 0
                result['usage']["total_tokens"] = 0
            result['usage']["prompt_tokens"] = em['data']['usage']["prompt_tokens"]
            result['usage']["total_tokens"] += em['data']['usage']["total_tokens"]

        return result
