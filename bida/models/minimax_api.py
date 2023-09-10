import json
from typing import (Any, Tuple)

from bida import Util
from bida.core.model_api_base import ModelAPIBase
from bida.core.llm_api_base import LLMAPIBase
from bida.core.embedding_api_base import EmbeddingAPIBase
from bida.models.minimax_sdk import minimax

class minimax_cc_api(LLMAPIBase, EmbeddingAPIBase):
    '''
    minimax的ChatCompletion API封装类，支持大模型和Embedding模型的调用。
    '''
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        minimax.group_id = ModelAPIBase.get_config_value(self.api_config["group_id"])
        minimax.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])
        minimax.chat_completions_baseurl = ModelAPIBase.get_config_value(self.api_config["chat_completions_baseurl"])
        minimax.embeddings_baseurl = ModelAPIBase.get_config_value(self.api_config["embeddings_baseurl"])
        
        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(minimax.api_key), 
            groupID=Util.mask_key(minimax.group_id),
        )
    
    ################################################
    #  completions和chatcompletions的能力          #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return True
    
    def get_completion_content(self, response) -> Any:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_chatcompletion_content(self, response) -> str:
        # 状态码(StatusCode)代表服务的返回状态，0为正确返回，所有的非零值均代表一类报错
        # https://api.minimax.chat/document/algorithm-concept?id=6433f37594878d408fc8295d
        if response['base_resp']['status_code'] != 0:
            raise Exception(f"生成文本出错，代码：{response['base_resp']['status_code']}，错误描述：{response['base_resp']['status_msg']}")
        finish_reason = response["choices"][0]["finish_reason"]
        if finish_reason == 'max_output':
            raise Exception('输入+模型输出内容超过模型能力限制')
        else:   # stop是正常的返回内容, length是超过tokens_to_generate限制所致
            result = response["choices"][0]["text"].strip()
                
        return result
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        # 状态码(StatusCode)代表服务的返回状态，0为正确返回，所有的非零值均代表一类报错
        # https://api.minimax.chat/document/algorithm-concept?id=6433f37594878d408fc8295d
        json_str = chunk.decode('utf-8')
        if not json_str.startswith("data"):
            if json_str != '' and json.loads(json_str).get("base_resp") is not None:
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
        elif finish_reason == 'stop': # stop时，会再次返回完整内容
            result=""
            fullcontent = chunk["choices"][0]["delta"]
        else: # length是超过tokens_to_generate限制所致，不报错
            result = chunk["choices"][0]["delta"]
            result = self.process_stream_data(stream_callback, result, stream_callback_args)
        return result, stop, fullcontent
    
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        if not isinstance(response, dict):
            json_str = response.decode('utf-8')
            json_str = json_str.split("data: ", 1)[1]
            response = json.loads(json_str)
        
        count = response["usage"]["total_tokens"]
            
        return 0, count
    
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
        if not prompt or prompt.replace("\n", " ").strip() == '':
            raise Exception("请输入有效的问题。")
        
        # 从chat_messages提取第一个做为system_prompt传进去
        system_prompt = None
        if chat_messages and chat_messages[0]['role'] == 'system':
            system_message = chat_messages.pop(0)
            system_prompt = system_message['content']
            if not system_prompt or system_prompt.replace("\n", " ").strip() == '':
                system_prompt = None
        # 转换chat_messages格式为minimax的格式标准：
        new_chat_messages = []
        for item in chat_messages:
            new_item = {}
            if item["role"] == "user":
                new_item["sender_type"] = "USER"
            elif item["role"] == "assistant":
                new_item["sender_type"] = "BOT"
            new_item["text"] = item["content"]
            new_chat_messages.append(new_item)
        # 将prompt添加到messages中
        new_chat_messages.append({"sender_type": "USER", "text": prompt})
        # 调用模型
        response = minimax.call(
            model=self.model_name,
            system_prompt=system_prompt,
            chat_messages=new_chat_messages,
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

    def embeddingcompletion(self, 
                            inputText,
                            *args, **kwargs
                            ):
        '''
        minimax的embeddings，有如下限制：每个文本长度不超过 4096个token。
        采用了query和db分离的算法方案，使用embedding_type指定生成类型，如果是要作为被检索文本: db，如果是作为检索文本: query。
        '''
        embedding_type = kwargs['embedding_type'] if 'embedding_type' in kwargs else None
        if embedding_type is None or embedding_type not in ('db', 'query'):
            raise Exception("请使用embedding_type指定生成类型，如果是要作为被检索文本: db，如果是作为检索文本: query")
        response = minimax.embeddings_call(
            input=inputText, 
            model=self.model_name,
            embedding_type=embedding_type
            )
        if response['base_resp']['status_code'] != 0:
            raise Exception(f"生成文本出错，代码：{response['base_resp']['status_code']}，错误描述：{response['base_resp']['status_msg']}")
        
        result = {}
        result['data'] = [{"index": i, "embedding": v} for i, v in enumerate(response["vectors"])]
        # 接口没有token信息
        result['usage'] = {"prompt_tokens": 0, "total_tokens": 0}
        
        return result
    