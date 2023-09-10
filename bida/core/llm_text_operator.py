from bida import Config
from bida import Util
from bida.core.llm_operator import LLMBase
from bida.core.prompt_template_base import PromptTemplateBase

class TextLLM(LLMBase):
    """
    实现“completions”的大模型操作
    """
    def generate_mode(self) -> Config.ModelGenerateModeType:
        '''模型运行的模式：completions'''
        return Config.ModelGenerateModeType.Completions
    
    def completion(
            self, 
            prompt, 
            prompt_template: PromptTemplateBase=None,
            stream_callback = None,
            stream_callback_args = None,
            *args, **kwargs
            ) -> str:
        """
        同步方式进行调用，传入问题内容，返回值是模型的答案，
        创建实例时传入的提示词模板，会自动调用，此处也可以替换为新的提示词模板。

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
            prompt, response = self._do_completion(
                prompt=prompt, 
                prompt_template=prompt_template, 
                instance=instance, 
                stream=stream, 
                *args, **kwargs
                )
            # 获取返回内容
            if stream:
                answer = ""
                chunk = None
                for chunk in response:
                    content, stop, fullcontent = instance.get_completion_stream_content(chunk, answer, stream_callback, stream_callback_args)
                    answer += content
                    if stop:
                        if fullcontent:
                            answer = fullcontent
                        break
                response = chunk
            else:
                answer = instance.get_completion_content(response)
            
            # 模型执行完成后的处理
            self._do_completed(stream, instance, prompt, answer, response)
            
            return answer
        except Exception as e:
            Util.log_error(e)
            raise e

    def acompletion(
            self, 
            prompt, 
            prompt_template: PromptTemplateBase=None,
            stream_callback = None,
            stream_callback_args = None,
            increment = False,
            *args, **kwargs
            ) -> str:
        """
        迭代方式进行chat，传入问题内容，用迭代器的方式返回每次的内容

        stream_callback: 如果需要流式处理返回结果，请定义流式处理函数：

            # 请参考下面的实现方式，data是每次返回的内容：

            def my_stream_process_data(data, stream_callback_args):         
            
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
            prompt, response = self._do_completion(
                prompt=prompt, 
                prompt_template=prompt_template, 
                instance=instance, 
                stream=stream, 
                *args, **kwargs
                )
            
            answer = ""
            chunk = None
            for chunk in response:
                content, stop, fullcontent = instance.get_completion_stream_content(chunk, answer, stream_callback, stream_callback_args)
                if content != '':
                    answer += content
                    if increment:
                        yield content
                    else:
                        yield answer
                if stop:
                    if fullcontent:
                        answer = fullcontent
                    break
            response = chunk

            # 模型执行完成后的处理
            self._do_completed(stream, instance, prompt, answer, response)
            
            return answer
        except Exception as e:
            Util.log_error(e)
            raise e

    def _do_completion(self, prompt, prompt_template, instance, stream, *args, **kwargs):
        # 根据提示词模板获取主提示词
        main_prompt = self.get_main_prompt_from_template(prompt_template)
        prompt = instance.merge_prompt(main_prompt, prompt)
        
        # 默认参数和传入参数的兼容性处理
        kwargs_copy, temperature_copy, max_tokens_copy = self.fix_default_params_with_kwargs(kwargs)

        # 模型调用前的log
        prefixstr = f"[{self.model_type}] completions:"
        self.log_begin(prefixstr)
        self.log(prefixstr, f"model_name={instance.model_name}")
        self.log(prefixstr, f"Prompt Length={len(prompt)}")

        # 调用模型的执行方法
        response = instance.textcompletion(
            prompt=prompt, 
            temperature=temperature_copy,
            max_tokens=max_tokens_copy,
            stream=stream,
            *args, **kwargs_copy
            )
        return prompt, response
    
    def _do_completed(self, stream, instance, prompt, answer, response):
        # 计算token数
        prompt_token_count, answer_token_count = instance.get_response_token_count(response, stream)
        prompt_token_count = prompt_token_count if prompt_token_count > 0 else instance.get_text_token_count(prompt)
        answer_token_count = answer_token_count if answer_token_count > 0 else instance.get_text_token_count(answer)

        # 模型调用完成后的log
        prefixstr = f"[{self.model_type}] completions:"
        self.log(prefixstr, f"Answer Length={len(answer)}")
        self.log(prefixstr, f"Prompt Tokens={prompt_token_count}, Answer Tokens={answer_token_count}, Total Tokens={prompt_token_count+answer_token_count}")
        self.log_end(prefixstr)
