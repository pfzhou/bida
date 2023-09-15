from bida.core.llm_api_base import LLMAPIBase
from bida import Util
from bida.core.model_api_base import ModelAPIBase
import sensenova
from typing import (
    Any,
    Tuple,
)


class sensetime_api(LLMAPIBase):
    '''
    商汤模型的chatCompletion API封装
    商汤提供：
    1. chatCompletion
    2. 基于会话的chatCompletion
    由于用户可以通过chatcompletion自行构建回话管理，本类只实现了chatCompletion
    商汤chatCompletion支持基于知识的问答，但是需要用户自行构建知识库，本类不实现知识库的创建和管理。
    商汤chatCompletion的API文档：https://platform.sensenova.cn/#/doc?path=/chat/ChatCompletions/ChatCompletions.md
    '''
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()
        sensenova.access_key_id = ModelAPIBase.get_config_value(self.api_config["access_key_id"])
        sensenova.secret_access_key = ModelAPIBase.get_config_value(self.api_config["secret_access_key"])

        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(sensenova.access_key_id), 
            secret_key=Util.mask_key(sensenova.secret_access_key), 
        )
    
    ################################################
    #  chatCompletions的能力                        #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return True
    
    def get_completion_content(self, response) -> str:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_chatcompletion_content(self, response) -> Any:
        finish_reason = response['data']["choices"][0]["finish_reason"]

        if finish_reason == 'sensitive':
            raise Exception('输入内容触发敏感词停止生成.')
        # 以下两种截断不报错    
        # elif finish_reason == 'length':   
        #     raise Exception('因达到最大生成长度停止生成.')
        # elif finish_reason == 'context':
        #     raise Exception('输入内容触发模型上下文长度限制.')
        else:
            result = response['data']["choices"][0]["message"].strip()
            return result
        
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        stop = False
        finish_reason = chunk['data']["choices"][0]["finish_reason"]

        if finish_reason == 'sensitive':
            raise Exception('输入内容触发敏感词停止生成.')
        # 以下两种截断不报错
        # elif finish_reason == 'length':
        #     raise Exception('因达到最大生成长度停止生成.')
        # elif finish_reason == 'context':
        #     raise Exception('输入内容触发模型上下文长度限制.')
        else: # "" 和 "stop" 都是正常的返回内容
            result = chunk['data']["choices"][0]["delta"].strip()
            result = self.process_stream_data(stream_callback, result, stream_callback_args)
        stop =  True if finish_reason == 'stop' or finish_reason == 'length' else False
        return result, stop, None
    
    def get_response_token_count(self, response, stream=False) -> Tuple[int, int]:
        prompt_tokens = response['data']["usage"]["prompt_tokens"]
        completion_tokens = response['data']["usage"]["completion_tokens"]
        return prompt_tokens, completion_tokens
    

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
        if temperature == 0.0:
            temperature = 0.01
        chat_messages.append({"role": 'user', "content": prompt})
        # 调用商汤模型推理服务
        # https://platform.sensenova.cn/#/doc?path=/chat/ChatCompletions/ChatCompletions.md
        resp = sensenova.ChatCompletion.create(
            messages=chat_messages,
            model=self.model_name,
            stream=stream,
            temperature=temperature,
            max_new_tokens=max_tokens,
            *args, **kwargs
        )
        
        return resp
    