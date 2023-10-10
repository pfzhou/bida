import json
from typing import Any, Tuple

from bida import Util
from bida.core.llm_api_base import LLMAPIBase
from bida.models.tencent_sdk import TencentHY
from bida.core.model_api_base import ModelAPIBase

class tencent_api(LLMAPIBase):
    '''
    腾讯混元模型的chatCompletion API封装，仅提供chatCompletion支持
    '''
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        TencentHY.app_id = int(ModelAPIBase.get_config_value(self.api_config["app_id"]))
        TencentHY.secret_id = ModelAPIBase.get_config_value(self.api_config["secret_id"])
        TencentHY.secret_key = ModelAPIBase.get_config_value(self.api_config["secret_key"])
        TencentHY.base_url = ModelAPIBase.get_config_value(self.api_config["base_url"])
        
        self.log_init(
            model_name=self.model_name, 
            app_id=Util.mask_key(str(TencentHY.app_id)),
            api_key=Util.mask_key(TencentHY.secret_id), 
            secret_key=Util.mask_key(TencentHY.secret_key), 
        )

    ################################################
    #  chatCompletions的能力                        #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return False

    def get_completion_content(self, response) -> Any:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_chatcompletion_content(self, response) -> str:
        if "error" in response:
            raise Exception(f"模型推理端报错，代码：{response['error']['code']}，错误描述：{response['error']['message']}")
        
        output_str = response['choices'][0]['messages']['content']
        return output_str
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        if chunk.data != '':
            data_js = json.loads(chunk.data)
            
            if 'error' in data_js:
                raise Exception(f"模型推理端报错，代码：{data_js['error']['code']}，错误描述：{data_js['error']['message']}")
            finish_reason = data_js['choices'][0]['finish_reason']
            content = data_js['choices'][0]['delta']['content']
            content = self.process_stream_data(stream_callback,content,stream_callback_args)
            isstop = True if finish_reason == 'stop' else False
            return content, isstop, None

    def get_response_token_count(self, response, stream=False) -> Tuple[int, int]:
        if stream:
            data = response.data
            data = json.loads(data)
        else:
            data = response
        prompt_tokens = data['usage']['prompt_tokens'] if data.get('usage') else 0
        answer_tokens = data['usage']['completion_tokens'] if data.get('usage') else 0
        return prompt_tokens, answer_tokens

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
        """
        腾讯混元接口文档：https://cloud.tencent.com/document/product/1729/97732
        输入message最大40轮历史对话，最大长度3000token，输出的回答最大长度1024tokens
        """
        chat_messages.append({"role": "user", "content": prompt})

        response = TencentHY.call(      
            messages=chat_messages,
            temperature=temperature,
            stream=stream,
            *args, **kwargs
        )
        return response