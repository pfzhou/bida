import json
from typing import (Any, Tuple)

from bida import Util
from bida.core.model_api_base import ModelAPIBase
from bida.models.baidu_sdk import baidu_wenxin
from bida.models.baidu_api import baidu_wenxin_api

class baidu_thirdmodels_api(baidu_wenxin_api):
    '''
    百度对托管的第三方模型的API封装类
    '''
    
    ################################################
    #  completions和chatcompletions的能力           #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    #                                              #
    #  从baidu_wenxin_api继承，大部分方法不再重复实现 #
    ################################################

    def chatcompletion(
            self,
            prompt,
            chat_messages,
            temperature,
            max_tokens,
            stream,
            *args, **kwargs
            ):
        # 百度托管的第三方模型仅支持历史消息和流式，其他参数都不支持
        response = baidu_wenxin.call(
            model=self.model_name,
            prompt=prompt,
            history=chat_messages,
            temperature=None,
            stream=stream,
            *args, **kwargs
            )
        return response
    