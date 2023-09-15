from http import HTTPStatus
from typing import (Any, Tuple)

from bida.core.model_api_base import ModelAPIBase
from bida.core.llm_api_base import LLMAPIBase
from bida.core.embedding_api_base import EmbeddingAPIBase
from bida import Util

import dashscope as ds

class aliyun_tongyiqianwen_api(LLMAPIBase, EmbeddingAPIBase):
    
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        super().init_Environment()

        ds.api_key = ModelAPIBase.get_config_value(self.api_config["api_key"])

        self.log_init(
            model_name=self.model_name, 
            api_key=Util.mask_key(ds.api_key), 
        )

    ################################################
    #  completions和chatcompletions的能力          #
    #  实现 LLMAPIBase 中 9个@abstractmethod 方法   #
    ################################################

    def support_system_prompt(self) -> bool:
        return True
    
    def get_completion_content(self, response) -> str:
        raise Exception("不支持Completion模式，请调用ChatCompletion")
    
    def get_completion_stream_content(self, response, previous_data, stream_callback, stream_callback_args) -> Tuple[str, bool, str]:
        raise Exception("不支持Completion模式，请调用ChatCompletion")

    def get_chatcompletion_content(self, response) -> Any:
        if response.status_code != HTTPStatus.OK:
            raise Exception(f"调用通义千问模型出错，错误码：{response.code}，状态：{response.status_code},错误信息：{response.message}")
        result = response["output"]['text'].strip()
        return result
    
    def get_chatcompletion_stream_content(self, chunk, previous_data, stream_callback, stream_callback_args) -> Tuple[Any, bool, Any]:
        """ ali模型每次都返回前面和本次的内容, 比如：
        ----------
        你好
        你好! 很高兴为你提
        你好! 很高兴为你提供帮助。有什么可
        你好! 很高兴为你提供帮助。有什么可以帮到你的吗?
        ----------
        finish_reason不一样，前面的全部是null，最后一次是stop
        """
        if chunk.status_code != HTTPStatus.OK:
            raise Exception(f"调用通义千问模型出错，错误码：{chunk.code}，状态：{chunk.status_code},错误信息：{chunk.message}")
        content = chunk["output"]["text"]

        '''有三种情况：正在生成时为null，\
            生成结束时如果由于停止token导致则为stop，\
            生成结束时如果因为生成长度过长导致则为length，因此不作为错误异常，正常输出。
        '''
        finish_reason = chunk["output"]["finish_reason"]
        stop = True if finish_reason == "stop" or finish_reason == "length" else False

        fullcontent = content if stop else None
        # 截取每次新增的内容
        if previous_data is not None and previous_data != '' and previous_data in content:
            content = content[len(previous_data):]
        if content != '':
            content = self.process_stream_data(stream_callback, content, stream_callback_args)
        
        return content, stop, fullcontent
    
    def get_response_token_count(self, response, stream) -> Tuple[int, int]:
        i = response["usage"].input_tokens
        j = response["usage"].output_tokens
        return i, j
    
    def get_text_token_count(self, text) -> int:
        # https://help.aliyun.com/zh/dashscope/product-overview/metering-and-billing-6?spm=a2c4g.11186623.0.0.5f4b5c7bUqNAAI
        # 1个token通常对应一个汉字；对于英文文本来说，1个token通常对应3至4个字母
        words_count, punctuation_count, chinese_count, other_characters_count = Util.calculate_text_elements_count(text)
        # 一个英文单词按评价5、6个字母计算，忽略标点符号，非汉字的其他语言字符按2个计算
        sum = chinese_count + words_count*1.5 + other_characters_count*2
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
        chat_messages.append({"role": 'user', "content": prompt})
        # 调用ali的对象, 不支持temperature和max_tokens，有max_length参数，可以在kwargs中传递进来
        response = ds.Generation.call(
            model=self.model_name,
            messages=chat_messages,
            stream=stream,
            *args, **kwargs
            )
        return response
    
    
    #######################################################
    #  embedding 向量生成的能力                            #
    #  实现 EmbeddingAPIBase 中1个@abstractmethod 方法     #
    #######################################################

    def embeddingcompletion(self, 
                            inputText,
                            *args, **kwargs
                            ):
        '''
        阿里的embeddings，有如下限制：\
        （1）最多支持10条; \
        （2）每个文本长度不超过 2048个token。 \
        本函数使用循环方式每次执行10个字符串，最后将所有内容拼成一个对象返回，其中：\
        embedding列表合并，index按生成顺序重新生成，tokens数据求和，其他字段用最后一次返回的内容。
        '''
        maxlen = 10
        listlen = len(inputText)
        result = []
        for i in range(0, listlen, maxlen):
            sublist = inputText[i:i+maxlen]
            
            subresponse = ds.TextEmbedding.call(input=sublist, 
                                                model=self.model_name
                                                )
            
            if subresponse.status_code != HTTPStatus.OK:
                raise Exception(f"生成embedding出错，代码：{subresponse.status_code}，错误描述：{subresponse.message}")
            
            if result == []:
                result = subresponse
                continue
            
            # 更新subresponse中的index值
            for item in subresponse.output['embeddings']:
                item['text_index'] += len(result.output['embeddings'])

            # 将subresponse的data字段加到result的data字段后面
            result.output['embeddings'].extend(subresponse.output['embeddings'])

            # 将subresponse的usage字段的值加到result的usage字段的值上
            result['usage']["total_tokens"] += subresponse['usage']["total_tokens"]

        # 修正结构为通用结构
        result['data'] = result['output'].pop('embeddings')
        for item in result['data']:
            item['index'] = item['text_index']
        result['usage']['prompt_tokens'] = result['usage']['total_tokens'] 

        return result
