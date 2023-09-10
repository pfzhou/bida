import json
from typing import (Any, Tuple)

from bida import Util
from bida.core.model_api_base import ModelAPIBase
from bida.core.llm_api_base import LLMAPIBase
from bida.core.embedding_api_base import EmbeddingAPIBase
from bida.models.baidu_sdk import baidu_wenxin

class baidu_wenxin_api(LLMAPIBase, EmbeddingAPIBase):
    '''
    百度文心一言的API封装类，支持LLM Chat和Embeddings的调用。
    '''
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        baidu_wenxin.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])
        baidu_wenxin.secret_key = ModelAPIBase.get_config_value(self.api_config["secret_key"])
        baidu_wenxin.access_token_url = ModelAPIBase.get_config_value(self.api_config["access_token_url"])
        model_baseurls = self.api_config["model_baseurls"]
        if model_baseurls is not None:
            for key, value in model_baseurls.items():
                baidu_wenxin.model_baseurls[key] = ModelAPIBase.get_config_value(value)
        
        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(baidu_wenxin.api_key), 
            secret_key=Util.mask_key(baidu_wenxin.secret_key),
        )
    ################################################
    #  completions和chatcompletions的能力          #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return False
    
    def get_completion_content(self, response) -> Any:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_completion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_chatcompletion_content(self, response) -> str:
        if response.get("error_code") is not None:
            raise Exception(f"生成文本出错，代码：{response['error_code']}，错误描述：{response['error_msg']}")
        result = response["result"].strip()
        return result
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        stop = False
        content = ''
        json_str = chunk.decode('utf-8')
        if not json_str.startswith("data"):
            if json_str != '' and json.loads(json_str).get("error_code") is not None:
                raise Exception(f"生成文本出错，代码：{json.loads(json_str).get('error_code')}，错误描述：{json.loads(json_str).get('error_msg')}")
        else:
            try:
                json_str = json_str.split("data: ", 1)[1]
            except:
                raise Exception(json_str)
        
            chunk = json.loads(json_str)
            content = chunk["result"]

            # 调用外部传递过来的处理函数处理回调返回的数据
            if content != '':
                content = self.process_stream_data(stream_callback, content, stream_callback_args)

            stop = chunk.get("is_end", False)
        return content, stop, None
    
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        if not isinstance(response, dict):
            json_str = response.decode('utf-8')
            json_str = json_str.split("data: ", 1)[1]
            response = json.loads(json_str)

        i = response["usage"]["prompt_tokens"]
        if stream:
            j = response["usage"]["total_tokens"] - i           # 流式下，每轮返回的是当轮的token数，所以用最后一轮的总数
        else:
            j = response["usage"]["completion_tokens"]
        return i, j
    
    def get_text_token_count(self, text) -> int:
        # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
        # token统计信息，token数 = 汉字数+单词数*1.3 （仅为估算逻辑）
        words_count, punctuation_count, chinese_count, other_characters_count = Util.calculate_text_elements_count(text)
        # 忽略标点符号，非汉字的其他语言字符按2个计算
        sum = chinese_count + words_count*1.3 + other_characters_count*2
        return sum
    
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
        # 调用baidu的对象，不支持max_tokens，会内部自己管理
        response = baidu_wenxin.call(
            model=self.model_name,
            prompt=prompt,
            history=chat_messages,
            temperature=temperature,
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
        百度的embeddings，有如下限制：
        （1）文本数量不超过16; 
        （2）每个文本长度不超过 384个token; 
        （3）输入文本不能为空，如果为空会报错。

        bida会剔除为空的内容不做转换，本函数使用循环方式每次执行16个字符串，最后将所有内容拼成一个对象返回，所以超过也不会报错。
        其中：embedding列表合并，index按生成顺序重新生成，tokens数据求和，其他字段用最后一次返回的内容。
        '''
        maxlen = 16
        listlen = len(inputText)
        result = []
        for i in range(0, listlen, maxlen):
            sublist = inputText[i:i+maxlen]
            subresponse = baidu_wenxin.embeddings_call(
                input=sublist, 
                model=self.model_name
                )
            if result == []:
                result = subresponse
                if result.get("error_code") is not None:
                    raise Exception(f"生成embedding出错，代码：{result['error_code']}，错误描述：{result['error_msg']}")
                continue
            
            # 更新subresponse中的index值
            for item in subresponse['data']:
                item['index'] += len(result['data'])

            # 将subresponse的data字段加到result的data字段后面
            result['data'].extend(subresponse['data'])

            # 将subresponse的usage字段的值加到result的usage字段的值上
            result['usage']["prompt_tokens"] += subresponse['usage']["prompt_tokens"]
            result['usage']["total_tokens"] += subresponse['usage']["total_tokens"]

        return result
    