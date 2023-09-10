from typing import (
    Any,
    Tuple,
)
import json

from bida.core.model_api_base import ModelAPIBase
from bida.core.llm_api_base import LLMAPIBase
from bida.core.embedding_api_base import EmbeddingAPIBase
from bida import Util
from bida import ManagementFunctions

import openai

class openai_api_v1(LLMAPIBase, EmbeddingAPIBase):
    
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        # 如果使用远端转发代理或本机代理，也需要读取配置信息
        # 一般来说，这两个配置是互斥的，就是一个配置值，另外一个就要删掉或设为''
        openai.api_base = ModelAPIBase.get_config_value(self.api_config["api_base"])
        openai.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])
        openai.proxy = ModelAPIBase.get_config_value(self.api_config["proxy"])
        openai.organization = ModelAPIBase.get_config_value(self.api_config["organization"])

        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(openai.api_key), 
            api_base=openai.api_base, 
            proxy=openai.proxy, 
            organization=openai.organization,
        )

    ################################################
    #  completions和chatcompletions的能力          #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return True
    
    def get_completion_content(self, response) -> str:
        result = response["choices"][0]["text"].strip()
        return result
    
    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        content = self.get_completion_content(chunk)
        # 调用外部传递过来的处理函数处理异步返回的数据
        content = self.process_stream_data(stream_callback, content, stream_callback_args)

        stop =  True if chunk["choices"][0]["finish_reason"] == 'stop' else False
        return content, stop, None
    
    def get_chatcompletion_content(self, response) -> Any:
        finish_reason = response["choices"][0]["finish_reason"]

        if finish_reason == 'content_filter':
            raise Exception('Omitted content due to a flag from our content filters.')
        elif finish_reason == 'function_call':  # 函数调用
            # 如果是函数调用，就返回调用对象
            result = response["choices"][0]["message"]
        # 超长经常是因为设定max_tokens导致，因此不作为错误抛异常，正常输出
        # elif finish_reason == 'length':
        #     raise Exception('生成长度过长导致出错。Incomplete model output due to max_tokens parameter or token limit.')
        else:           # stop or null, 都是正常的返回内容
            result = response["choices"][0]["message"]["content"].strip()
                
        return result
    
    _function_call_name=''
    _function_call_arguments = ''
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        """chat模式的流式下，获取每次返回的内容，各模型可以override这个方法。"""
        def save_function_call_info(func_content):
            if func_content.get('name'):
                self._function_call_name = func_content['name']
                self._function_call_arguments = ''
            self._function_call_arguments += func_content.get('arguments', '')

        stop = False
        finish_reason = chunk["choices"][0]["finish_reason"]
        if finish_reason == 'content_filter':
            raise Exception('Omitted content due to a flag from our content filters.')
        elif finish_reason == 'function_call':  # 函数调用
            # 如果是函数调用，就返回调用对象
            result = {
                        "content": None,
                        "function_call": {
                                            "name": self._function_call_name,
                                            "arguments": self._function_call_arguments
                                        },
                        "role": 'assistant'
                    }
            stop = True
            # 函数参数生成完之后，stream输出信息
            result = self.process_stream_data(stream_callback, result, stream_callback_args)
        # 超长经常是因为设定max_tokens导致，因此不作为错误异常，正常输出
        # elif finish_reason == 'length':  
        #     raise Exception('生成长度过长导致出错。Incomplete model output due to max_tokens parameter or token limit.')
        else: # stop or null, 都是正常的返回内容
            if chunk["choices"][0]["delta"].get("function_call",""):
                result = chunk["choices"][0]["delta"]['function_call']
                save_function_call_info(result)
                # 函数参数没有生成完之前，stream不输出信息
            else:
                result = chunk["choices"][0]["delta"].get("content","")
                # 调用外部传递过来的处理函数处理异步返回的数据
                result = self.process_stream_data(stream_callback, result, stream_callback_args)
            stop =  True if finish_reason == 'stop' or finish_reason == 'length' else False
        return result, stop, None
    
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        if stream:      # 流模式下无法获取token数，只能使用text计算token数，因此返回0
            i = 0
            j = 0
        else:
            i = response["usage"]["prompt_tokens"]
            j = response["usage"]["completion_tokens"]
        return i, j
    
    def get_text_token_count(self, text) -> int:
        return Util.get_text_token_count(text=text)
    
    def textcompletion(
            self, 
            prompt, 
            temperature, 
            max_tokens, 
            stream, 
            *args, **kwargs
            ):
        # 调用openai的对象
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            *args, **kwargs
            )
        return response
    
    def chatcompletion(
            self,
            prompt,
            chat_messages,
            temperature,
            max_tokens,
            stream,
            *args, **kwargs
            ):
        if isinstance(prompt, str):
            chat_messages.append({"role": 'user', "content": prompt})
        else:   #function_call
            assert prompt['role'] == 'function', "未知的调用结果"
            chat_messages.append(prompt)
        # 调用openai的对象
        response = openai.ChatCompletion.create(
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
    #  实现 EmbeddingAPIBase 中1个@abstractmethod 方法     #
    #######################################################

    def embeddingcompletion(
            self, 
            inputText,
            *args, **kwargs
            ):
        response = openai.Embedding.create(
            input=inputText, 
            model=self.model_name,
            )
        return response
    
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
            ):
        """执行函数调用然后将结果再次调用大模型获得最终结果"""

        # 调用函数并拿到返回结果
        func_result = self._do_function_call(func_response)
        
        
        # 再次调用大模型对函数返回结果进行处理
        if is_iter:
            result = llm_operator.achat(
                prompt=func_result,
                stream_callback=stream_callback,
                increment=increment,
                *args, **kwargs
                )
        else:
            result = llm_operator.chat(
                prompt=func_result,
                stream_callback=stream_callback,
                *args, **kwargs
                )
        return result

    def _do_function_call(self, response_message) -> dict:
        """根据生成的function call调用信息，使用指定的function进行查询并返回答案"""
        # 可用函数列表
        available_functions = ManagementFunctions.get_available_function_names()
        # 获取函数名称
        function_name = response_message["function_call"]["name"]
        # 获取函数对象
        fuction_to_call = available_functions[function_name]
        # 获取函数参数
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 调用函数获取数据
        function_response = fuction_to_call(**function_args)
        # 返回结果对象
        result = {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
        return result