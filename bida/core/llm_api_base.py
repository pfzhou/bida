from abc import ABC, abstractmethod
from typing import (Any, Tuple)

from bida.core.model_api_base import ModelAPIBase

class LLMAPIBase(ModelAPIBase, ABC):
    """
    completions和chatcompletions两类能力模型API的基类, 
    定义了接入模型API需要实现的抽象方法和通用能力。
    """

    max_tokens: int = None
    """模型的max_tokens值"""
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()
        self.max_tokens = self.model_config.get("max_tokens", 0)

    ################################################
    #  如果要实现completions和chatcompletions的能力 #
    #  以下@abstractmethod 9 个方法必须在子类中实现  #
    ################################################
    
    @abstractmethod
    def support_system_prompt(self) -> bool:
        '''是否支持system类型的提示词，在最头部每次都提交给LLM，例如：openai支持，百度、阿里不支持'''
    
    @abstractmethod
    def get_completion_content(self, response) -> str:
        '''在completion时，从返回对象中提取返回内容'''

    @abstractmethod
    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        '''
        completion模式的流式下，获取每次返回的内容，各模型如果要支持completion模式，必须实现这个方法。
        
        三个返回参数：1、每次chunk的内容; 2、是否结束标志; 3、结束时，返回这次GC的完整内容，如果无法返回完整内容，请返回None。
        '''
        
    @abstractmethod
    def get_chatcompletion_content(self, response) -> Any:
        '''在chat completion时，从返回对象中提取返回内容'''

    @abstractmethod
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        '''
        chat模式的流式下，获取每次返回的内容，各模型如果要支持chat completion模式，必须实现这个方法。
        
        三个返回参数：1、每次chunk的内容; 2、是否结束标志; 3、结束时，返回这次GC的完整内容，如果无法返回完整内容，请返回None。
        '''
    
    @abstractmethod
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        '''根据返回的response获取对应token数，返回参数：promt_token,answer_token'''
    
    @abstractmethod
    def get_text_token_count(self, text) -> int:
        '''根据text内容获取token数'''

    @abstractmethod
    def textcompletion(
        self, 
        prompt, 
        temperature, 
        max_tokens, 
        stream, 
        *args, **kwargs
        ):
        """各模型调用completion模式时具体实现"""
    

    @abstractmethod
    def chatcompletion(
        self,
        prompt,
        chat_messages,
        temperature,
        max_tokens,
        stream,
        *args, **kwargs
        ):
        """各模型调用chat completion模式时具体实现"""
    

    ##############################################
    #  以上的方法必须在子类中实现                  #
    #  以下的根据需要也可以在子类中override        #
    ##############################################

    def process_stream_data(self, stream_callback, data, stream_callback_args):
        """流式时，对返回内容的处理方法"""
        if stream_callback:
            if stream_callback_args:
                back_data = stream_callback(data, stream_callback_args)
            else:
                back_data = stream_callback(data)
            if back_data:
                data = back_data
        return data
            
    def merge_prompt(self, main_prompt, prompt):
        """将主prompt模板和当前prompt合并，各模型可以override这个方法。"""
        if main_prompt: 
            main_prompt += "\n" 
        return main_prompt + prompt

    ################################################
    #  如果要实现function_call的功能                #
    #  必须在子类中实现下面的这个方法                #
    ################################################

    def function_call(
            self,
            is_iter, 
            llm_operator, 
            func_response, 
            stream_callback,
            increment=None,
            *args, **kwargs
            ) -> dict:
        """执行函数调用然后将结果再次调用大模型获得最终结果"""
        raise Exception('当前模型不支持function call')
    