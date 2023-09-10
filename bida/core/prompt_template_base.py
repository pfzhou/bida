from abc import ABC, abstractmethod
from pydantic import Field
from typing import (
    Any,
    Callable,
    Mapping,
    Union,
    )

class PromptTemplateBase(ABC):
    '''
    模板的基类，继承是必须实现：main_prompt 主提示词,
    可以根据不同模型设置不一样的提示词，子模型model_name > 模型类型model_type > main_prompt。
    需要替换的关键词必须格式为【"{{{{关键字}}}}"(四个大括号)】，【区分大小写且不能有空格】。
    '''

    keyword_prefix = "{{{{"
    keyword_suffix = "}}}}"

    context_variables: Mapping[str, Union[str, Callable[[], str]]] = Field(
        default_factory=dict
    )

    model_function_registry = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.context_variables = kwargs

    @property
    @abstractmethod
    def main_prompt(self) -> str:
        return None

    @classmethod
    def using_model(cls, model_type, model_name=None):
        def decorator(func):
            classname = func.__qualname__.split('.', 1)[0]
            if classname not in cls.model_function_registry:
                cls.model_function_registry[classname] = {}
            if model_type not in cls.model_function_registry[classname]:
                cls.model_function_registry[classname][model_type] = []
            func_info_list = cls.model_function_registry[classname][model_type]
            for func_info in func_info_list:
                if model_name == func_info[1]:
                    raise Exception(f"在class: {classname} 中注册重复：model_type={model_type}, model_name={model_name}")
            cls.model_function_registry[classname][model_type].append((func, model_name))
            return func
        return decorator

    def implement_functionality(self, model):
        """按模型设定不同的提示词"""
        m_func = None
        classname = self.__class__.__name__
        if classname in self.model_function_registry:
            model_type = model.model_type.lower()
            if model_type in self.model_function_registry[classname]:
                func_info_list = self.model_function_registry[classname][model_type]
                for func_info in func_info_list:
                    func = func_info[0]
                    model_name = func_info[1]
                    # 如果还指定了model_name，判断当前model是否匹配
                    # 注意，这里不支持alias，必须使用完全匹配的名字，但不区分大小写
                    if model_name is None:
                        m_func = func
                    else:
                        if model_name.lower() == model.model_name.lower():
                            m_func = func
                            break

        return None if m_func is None else m_func(self)
    
        
    def format(self, model, **kwargs: Any) -> str:
        '''
        model是调用模板的Model对象；
        根据参数kwargs将主提示词模板中的内容进行替换并返回。
        如果有设置，会使用不同模型的主提示词，同时会合并实例创建时传入的参数一并替换。
        '''
        custom_prompt = self.implement_functionality(model)
        result = self.main_prompt if custom_prompt is None else custom_prompt
        if result == "" or result == None:
            return ""
        variables = {**self.context_variables, **kwargs}
        for key, value in variables.items():
            valuestr = ""
            if value:
                valuestr = value
            result = result.replace(
                f"{self.keyword_prefix}{key}{self.keyword_suffix}", 
                valuestr
                )
        return result

    def hasKeyword(self, keyword):
        result = False
        template = self.main_prompt
        if template and template != "":
            if template.find(f"{self.keyword_prefix}{keyword}{self.keyword_suffix}") >= 0:
                result = True
        return result

    