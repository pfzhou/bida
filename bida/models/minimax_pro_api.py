import json
from typing import (Any, Tuple)

from bida.core.llm_api_base import LLMAPIBase
from bida.models.minimax_sdk import minimax
from bida.models.minimax_api import minimax_cc_api
from bida import ManagementFunctions

class minimax_ccp_api(minimax_cc_api):
    '''
    minimax的ChatCompletionPro API封装类，支持大模型和Embedding模型的调用。
    '''
    
    ################################################
    #  completions和chatcompletions的能力          #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    #                                              #
    #  从minimax_cc_api继承，部分方法不再重复实现    #
    ################################################

    def support_system_prompt(self) -> bool:
        return False
    
    def get_chatcompletion_content(self, response) -> str:
        # 状态码(StatusCode)代表服务的返回状态，0为正确返回，所有的非零值均代表一类报错
        # https://api.minimax.chat/document/algorithm-concept?id=6433f37594878d408fc8295d
        if response['base_resp']['status_code'] != 0:
            raise Exception(f"生成文本出错，代码：{response['base_resp']['status_code']}，错误描述：{response['base_resp']['status_msg']}")
        finish_reason = response["choices"][0]["finish_reason"]
        if finish_reason == 'max_output':
            raise Exception('输入+模型输出内容超过模型能力限制')
        else:   # stop是正常的返回内容, length是超过tokens_to_generate限制所致
            if response["choices"][0]['messages'][0].get('function_call'):
                # 函数调用
                function_call = response["choices"][0]['messages'][0]['function_call']
                function_call['sender_name'] = response["choices"][0]["messages"][0]["sender_name"]
                result = {"function_call": function_call}
            else:
                result = minimax.MessageStrWrapper(
                    text=response["choices"][0]["messages"][0]["text"], 
                    sender_name=response["choices"][0]["messages"][0]["sender_name"]
                    )
                
        return result
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        # 状态码(StatusCode)代表服务的返回状态，0为正确返回，所有的非零值均代表一类报错
        # https://api.minimax.chat/document/algorithm-concept?id=6433f37594878d408fc8295d
        json_str = chunk.decode('utf-8')
        if not json_str:
            return '', False, None
        if not json_str.startswith("data"):
            if json.loads(json_str).get("base_resp") is not None:
                error = json.loads(json_str)['base_resp']
                raise Exception(f"生成文本出错，代码：{error['status_code']}，错误描述：{error['status_msg']}")
        else:
            try:
                json_str = json_str.split("data: ", 1)[1]
            except:
                raise Exception(json_str)
        
            chunk = json.loads(json_str)
        
        if chunk.get('base_resp') is not None and chunk['base_resp']['status_code'] != 0:
            raise Exception(f"生成文本出错，代码：{chunk['base_resp']['status_code']}，错误描述：{chunk['base_resp']['status_msg']}")
        finish_reason = chunk["choices"][0]["finish_reason"] if chunk["choices"][0].get("finish_reason") else None
        stop = True if finish_reason else False
        fullcontent = None

        if finish_reason == 'max_output':
            raise Exception('输入+模型输出内容超过模型能力限制')
        elif finish_reason == 'stop': # stop时，会再次返回完整内容chunk["choices"][0]["messages"][0]
            result=""
            if chunk["choices"][0]['messages'][0].get('function_call'):
                # 函数调用
                function_call = chunk["choices"][0]['messages'][0]['function_call']
                function_call['sender_name'] = chunk["choices"][0]["messages"][0]["sender_name"]
                result = fullcontent = {"function_call": function_call}
            else:
                fullcontent = minimax.MessageStrWrapper(text=chunk["choices"][0]["messages"][0]["text"], 
                                                        sender_name=chunk["choices"][0]["messages"][0]["sender_name"]
                                                    )
        else:  # length是超过tokens_to_generate限制所致，不报错
            if chunk["choices"][0]['messages'][0].get('function_call'):
                result = chunk["choices"][0]['messages'][0]['function_call'] #json.dumps(chunk["choices"][0]['messages'][0]['function_call'], ensure_ascii=False, indent=4)
            else:
                result = chunk["choices"][0]["messages"][0]['text']
            result = self.process_stream_data(stream_callback, result, stream_callback_args)
        return result, stop, fullcontent
    
    def chatcompletion(
            self,
            prompt,
            chat_messages,
            temperature,
            max_tokens,
            stream,
            *args, **kwargs
            ):
        # 转换chat_messages格式为minimax ccp的格式标准：
        new_chat_messages = []
        for item in chat_messages:
            new_item = {}
            if item["role"] == "user":
                new_item["sender_type"] = "USER"
            elif item["role"] == "assistant":
                new_item["sender_type"] = "BOT"
            elif item["role"] == "function":
                new_item["sender_type"] = "FUNCTION"
            else:
                raise Exception(f"[{item['role']}]未知的role类型")
            
            if item["content"] is None: # functioncall的content为None
                new_item["sender_name"] = item['function_call']["sender_name"]
                new_item["function_call"] = item['function_call'].copy()
                new_item["function_call"].pop("sender_name")
            else:
                textobj = json.loads(item["content"])
                new_item["sender_name"] = textobj["sender_name"]
                new_item["text"] = textobj["text"]
            new_chat_messages.append(new_item)
        
        # 将prompt添加到messages中
        if isinstance(prompt, str):
            message = json.loads(prompt)
            message["sender_type"] = "USER"
        else:   #function_call
            assert prompt['role'] == 'function', "未知的调用结果"
            content = json.loads(prompt["content"])
            message = {}
            message["sender_type"] = "FUNCTION"
            message["sender_name"] = content["sender_name"]
            message["text"] = content["text"]
        new_chat_messages.append(message)

        # 判断有无指定bot_setting和reply_constraints，没有就添加默认的参数
        if not kwargs.get("bot_setting"):
            kwargs["bot_setting"] = minimax.Default_Bot_Setting
        if not kwargs.get("reply_constraints"):
            kwargs["reply_constraints"] = minimax.Default_Reply_Constraints

        # 调用模型
        response = minimax.call(
            model=self.model_name,
            system_prompt=None,             # ccp模式下不支持system prompt，使用bot_setting.content
            chat_messages=new_chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            *args, **kwargs
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
        function_response = minimax.MessageStrWrapper(
            text=function_response, 
            sender_name=response_message["function_call"]["sender_name"]
            )
        
        result = {
            "role": "function",
            "name": function_name,
            "content": function_response,
            }
        return result


    