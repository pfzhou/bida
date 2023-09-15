import json

from bida.core.llm_api_base import LLMAPIBase
from bida.models.baichuan_sdk import baichuan
from bida.core.model_api_base import ModelAPIBase
from bida import Util
from typing import Any, Tuple

class baichuan_api(LLMAPIBase):
    '''
    百川模型的chatCompletion API封装，仅支持chatCompletion
    '''
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        baichuan.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])
        baichuan.secret_key = ModelAPIBase.get_config_value(self.api_config["secret_key"])
        baichuan.api_url = ModelAPIBase.get_config_value(self.api_config["api_url"])
        baichuan.api_url_stream = ModelAPIBase.get_config_value(self.api_config["api_url_stream"])

        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(baichuan.api_key), 
            secret_key=Util.mask_key(baichuan.secret_key), 
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
        if response.get("code") != 0:
            raise Exception(f"模型推理端报错，代码：{response['code']}，错误描述：{response['msg']}")
        finish_reason = response['data']['messages'][0]['finish_reason']
        result = response["data"]['messages'][0]['content']
        if finish_reason == 'stop':
            return result
        else:
            raise Exception(f"模型推理会话终止异常，finish_reason={finish_reason},content={result}")

    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        json_str = chunk.decode('utf-8')
        chunk = json.loads(json_str)
        if chunk.get("code") != 0:
            raise Exception(f"模型推理端报错，代码：{chunk['code']}，错误描述：{chunk['msg']}")
        finish_reason = chunk['data']['messages'][0]['finish_reason']
        content = chunk['data']['messages'][0]['content']
        content = self.process_stream_data(stream_callback,content,stream_callback_args)
        isstop = True if finish_reason == 'stop' else False
        return content, isstop, None

    def get_response_token_count(self, response, stream=False) -> Tuple[int, int]:
        if stream:
            return  0,0
        prompt_tokens = response['usage']['prompt_tokens'] if response.get('usage') else 0
        answer_tokens = response['usage']['answer_tokens'] if response.get('usage') else 0
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
        百川模型接口文档：https://platform.baichuan-ai.com/docs/api
        暂不支持temperature等模型超参数设置功能。
        """
        chat_messages.append({"role": "user", "content": prompt})
        response = baichuan.call(
            model=self.model_name,
            chatMessages=chat_messages,
            stream=stream,
            *args, **kwargs
        )
        return response