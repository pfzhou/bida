from typing import (
    Any,
    List,
    Optional,
    )
from datetime import datetime

from bida import Config
from bida import Util
from bida.core.llm_operator import LLMBase
from bida.core.conversation import Conversation, Message, MessageCategory, MessageStatus
from bida.core.prompt_template_base import PromptTemplateBase
from bida.core.conversation_buffer import BufferControllerBase

class ChatLLM(LLMBase):
    """
    实现“chat completions”的大模型操作
    """

    def __init__(
            self, 
            model_type, 
            model_name:str = None,
            prompt_template:PromptTemplateBase = None,
            temperature = 0.0,
            max_tokens = 1024, 
            stream_callback = None,
            auto_save_conversation = False,
            buffer_controller:BufferControllerBase = None,
            **model_kwargs: Any
            ):
        """
        创建大模型处理的实例
        """
        super().__init__(
            model_type=model_type, 
            model_name=model_name, 
            prompt_template=prompt_template, 
            temperature=temperature,
            max_tokens=max_tokens,
            stream_callback=stream_callback,
            **model_kwargs
            )
        # 初始化当前会话管理对象
        self.conversation = Conversation(self.model_type)
        self._auto_save_conversation = auto_save_conversation
        self._buffer_controller = buffer_controller


    def generate_mode(self) -> Config.ModelGenerateModeType:
        '''模型运行的模式：chat_completions'''
        return Config.ModelGenerateModeType.ChatCompletions
    
    _conversation: Conversation = None
    """会话记录"""
    _auto_save_conversation: bool = False
    """是否自动保存会话记录"""
    _buffer_controller: BufferControllerBase = None 
    chat_messages: Optional[List] = None
    """交互历史记录"""
    
    @property
    def conversation(self) -> Conversation:
        '''会话管理对象'''
        return self._conversation
    
    @conversation.setter
    def conversation(self, value: Conversation):
        if value is None or isinstance(value, Conversation):
            self._conversation = value
        else:
            raise ValueError("Value must be an instance of Conversation")

    def chat(
            self, 
            prompt, 
            stream_callback = None,
            stream_callback_args = None,
            *args, **kwargs
        ) -> str:
        """
        同步进行chat，传入问题内容，返回值是模型的答案

        stream_callback: 如果需要流式处理返回结果，请定义流式处理函数：

            # 请参考下面的实现方式，data是每次返回的内容：

            def my_stream_process_data(data, stream_callback_args):         
            
                print(data, end="", flush=True)    

        stream_callback_args: 回调函数的参数   
        """
        try:
            # 判断是否是流式处理
            stream_callback = stream_callback or self.stream_callback
            stream = True if stream_callback else False
            
            # 获取模型的API运行实例
            instance = self._get_model_api_instance()

            # 执行模型的调用
            prompt, response = self._do_chat(
                prompt,
                instance,
                stream,
                *args, **kwargs
                )
            if stream:
                answer = ""
                chunk = None
                # zhipuai use sseclient
                _response = response.events() if hasattr(response, 'events') else response
                for chunk in _response:
                    content, stop, fullcontent = instance.get_chatcompletion_stream_content(chunk, answer, stream_callback, stream_callback_args)
                    if content:
                        if isinstance(content, str):
                            answer += content
                        else:   # function_call
                            answer = content
                    if stop:
                        if fullcontent:
                            answer = fullcontent
                        break
                response = chunk
            else:
                answer = instance.get_chatcompletion_content(response)

            # 模型执行完成后的处理
            self._do_chat_completed(stream, instance, prompt, answer, response)

            # 处理function_call
            if not isinstance(answer, str):
                # 调用实例的函数运行方法，如果有函数调用，只会返回最终结果，回调函数可以获取过程信息
                answer = instance.function_call(is_iter=False,
                                                llm_operator=self, 
                                                func_response=answer, 
                                                stream_callback=stream_callback,
                                                *args, **kwargs
                                                )

            return answer
        except Exception as e:
            Util.log_error(e)
            raise e

    def achat(
            self, 
            prompt, 
            stream_callback = None,
            stream_callback_args = None,
            increment = False,
            *args, **kwargs
            ) -> str:
        """
        迭代方式进行chat，传入问题内容，用迭代器的方式返回每次的内容

        stream_callback: 如果需要流式处理返回结果，请定义流式处理函数：

            # 请参考下面的实现方式，data是每次返回的内容：

            def my_stream_process_data(data,  stream_callback_args):         
            
                print(data, end="", flush=True)   

        stream_callback_args: 回调函数的参数    

        increment: 返回内容是否是增量模式, 例如：

            增量模式：第一次返回：'hello '，第二次返回：'world', 

            全量模式：第一次返回：'hello '，第二次返回：'hello world' 
        """
        try:
            # 判断是否是流式处理
            stream_callback = stream_callback or self.stream_callback
            stream = True
            
            # 获取模型的API运行实例
            instance = self._get_model_api_instance()

            # 执行模型的调用
            prompt, response = self._do_chat(
                prompt,
                instance,
                stream,
                *args, **kwargs
                )

            answer = ""
            chunk = None
            # zhipuai use sseclient
            _response = response.events() if hasattr(response, 'events') else response
            for chunk in _response:
                content, stop, fullcontent = instance.get_chatcompletion_stream_content(chunk, answer, stream_callback, stream_callback_args)
                if content:
                    if isinstance(content, str):
                        answer += content
                        if increment:
                            yield content
                        else:
                            yield answer
                    else:   # function_call
                        answer = content
                        
                if stop:
                    if fullcontent:
                        answer = fullcontent
                    if not isinstance(answer, str):
                        yield str(answer)
                    break
                
            response = chunk

            # 模型执行完成后的处理
            self._do_chat_completed(stream, instance, prompt, answer, response)

            # 处理function_call
            if not isinstance(answer, str):
                # 调用实例的函数运行方法，这里必须要用迭代器
                for answer in instance.function_call(is_iter=True,
                                                llm_operator=self, 
                                                func_response=answer, 
                                                stream_callback=stream_callback,
                                                increment=increment,
                                                *args, **kwargs
                                                ):
                    yield str(answer)

            return answer
        except Exception as e:
            Util.log_error(e)
            raise e
        
    def _do_chat(self, prompt, instance, stream, *args, **kwargs):
        # 如果是第一次交互，根据提示词模板获取主提示词，并根据模型的特性判断是否要设定system prompt
        original_prompt = prompt
        if len(self.conversation) == 0:
            main_prompt = self.get_main_prompt_from_template(None)
            if instance.support_system_prompt() and main_prompt:
                self.conversation.append_message(
                    role='system', 
                    content=main_prompt,
                    tokens=instance.get_text_token_count(main_prompt),
                    )
            else:
                prompt = instance.merge_prompt(main_prompt, prompt)

        # 使用buffer控制类收缩会话历史（只影响提交给LLM的，不影响显示和保存）
        if self._buffer_controller:
            self._buffer_controller.contraction(self.conversation)

        # 给会话添加本次prompt的内容
        if isinstance(prompt, str):
            self.conversation.append_message(
                role='user', 
                content=prompt,
                display_content=original_prompt,
                status=MessageStatus.processing,
                )
        else:   # function call
            assert prompt['role'] == 'function', "未知的调用结果"
            self.conversation.append_message(
                role=prompt['role'], 
                content=prompt['content'], 
                name=str(prompt['name']),
                display_content=f"[{prompt['name']}]检索返回的结果处理中......",
                status=MessageStatus.processing,
                ) 
        # 保存到数据库
        if self._auto_save_conversation:
            self.conversation.persist()

        # 默认参数和传入参数的兼容性处理
        kwargs_copy, temperature_copy, max_tokens_copy = self.fix_default_params_with_kwargs(kwargs)
        # 生成message history
        history = Conversation.convert_history_to_llm_general_format(self.conversation.history_for_llm)
        
        # 模型调用前的log
        prefixstr = f"[{self.model_type}] chat_completions:"
        self.log_begin(prefixstr)
        self.log(prefixstr, f"model_name={instance.model_name}")
        self.log(prefixstr, f"Prompt Length={len(prompt)}")

        # 调用模型的执行方法
        response = instance.chatcompletion(
            prompt=prompt,
            chat_messages=history[:],
            temperature=temperature_copy,
            max_tokens=max_tokens_copy,
            stream=stream,
            *args, **kwargs_copy
            )
        
        # 给会话初始化answer的内容
        self.conversation.append_message(
            role='assistant', 
            content=None, 
            status=MessageStatus.processing,
            )    

        # 保存到数据库
        if self._auto_save_conversation:
            self.conversation.persist()

        return prompt, response

    def _do_chat_completed(self, stream, instance, prompt, answer, response):
        # 计算token数
        response_prompt_token_count, response_answer_token_count = instance.get_response_token_count(response, stream)
        
        # 转换会话中的消息状态和属性
        # prompt message
        prompt_message = self.conversation[-2] 
        assert prompt_message.status == MessageStatus.processing, '错误的消息状态'
        # 获取当次prompt的token数
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        if response_prompt_token_count == 0:
            response_prompt_token_count = instance.get_text_token_count(str(prompt))
        prompt_message.tokens = response_prompt_token_count
        prompt_message.status = MessageStatus.completed

        # response(answer) message
        answer_message = self.conversation[-1] 
        assert answer_message.status == MessageStatus.processing, '错误的消息状态'
        if isinstance(answer, str):
            answer_message.content = answer
            answer_message.display_content = answer
        else:   # function call
            assert answer.get('function_call'), "未知的调用结果"
            answer_message.content = None
            answer_message.function_call = answer['function_call']
            answer_message.display_content = f"使用[{answer['function_call']['name']}]检索中......"
        # 获取当次返回内容的token数
        if response_answer_token_count == 0:
            response_answer_token_count = instance.get_text_token_count(str(answer))
        answer_message.tokens = response_answer_token_count
        answer_message.end_time = datetime.now()
        answer_message.status = MessageStatus.completed
        
        # 保存到数据库
        if self._auto_save_conversation:
            self.conversation.persist()

        # 模型调用完成后的log
        prefixstr = f"[{self.model_type}] chat_completions:"
        self.log(prefixstr, f"Answer Length={len(answer)}")
        self.log(prefixstr, f"Prompt Tokens={response_prompt_token_count}, Answer Tokens={response_answer_token_count}, Total Tokens={response_prompt_token_count+response_answer_token_count}")
        self.log_end(prefixstr)
