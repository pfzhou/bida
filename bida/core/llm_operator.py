from typing import (
    Any,
    Optional,
)

from bida.core.model_operator_base import ModelOperatorBase
from bida.core.prompt_template_base import PromptTemplateBase

class LLMBase(ModelOperatorBase):
    '''
    操作生成型大语言模型LLM的基类
    '''
    temperature: float = 0.0
    """temperature调节输出的稳定性"""
    max_tokens: Optional[int] = None
    """每次生成返回的最大字符串"""
    stream_callback = None
    """
    如果需要流式处理返回结果，请定义流式处理函数：\
    # 请参考下面的实现方式，data是每次返回的内容：\
    def my_stream_process_data(data):         \
        print(data, end="", flush=True)
    """
    
    def __init__(
            self, 
            model_type, 
            model_name:str = None,
            prompt_template:PromptTemplateBase = None,
            temperature = 0.0,
            max_tokens = 1024, 
            stream_callback = None,
            **model_kwargs: Any
            ):
        """
        创建大模型处理的实例
        """
        super().__init__(
            model_type=model_type, 
            model_name=model_name, 
            prompt_template=prompt_template, 
            **model_kwargs
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream_callback =stream_callback

    @staticmethod
    def stream_callback_func(data):
        '''默认处理流式输出的函数，仅显示每次返回的内容，不做其他处理'''          
        print(data, end="", flush=True)

    def fix_default_params_with_kwargs(self, kwargs):
        '''
        因为temperature和max_tokens两个参数是Operator对象初始化时设置的默认值，已经添加到模型参数内，
        如果仍然向模型直接传递这两个参数就会重复导致报错，这里会做排重处理。

        同时，为了不影响原始传递进来的参数列表和Operator的默认值，向模型传递时全部做副本处理。
        '''
        kwargs_copy = kwargs.copy()
        temperature_copy = self.temperature
        max_tokens_copy = self.max_tokens
        if 'temperature' in kwargs_copy:
            temperature_copy = kwargs_copy.pop('temperature')
        if 'max_tokens' in kwargs_copy:
            max_tokens_copy = kwargs_copy.pop('max_tokens')
        return kwargs_copy,temperature_copy,max_tokens_copy