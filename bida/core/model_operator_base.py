from abc import ABC, abstractmethod
from pydantic import Field
from typing import (
    Any,
    Dict,
)
from datetime import datetime
import uuid

from bida import Util
from bida import Config
from bida import ManagementModels
from bida.core.model_api_base import ModelAPIBase
from bida.core.prompt_template_base import PromptTemplateBase

class ModelOperatorBase(ABC):
    '''
    模型操作基类
    '''
    model_type_config = None
    """模型类型的配置信息"""
    model_config = None
    """模型的配置信息"""
    api_config: dict = None
    """api的配置信息"""
    
    model_type: str
    """模型分类名称，非运行的模型，是模型的大类：Openai、Vicuna、ChatGLM等"""
    api_protocol: str
    """模型使用的协议，根据协议定义会调用不同的处理方法，默认会更倾向于遵循OpenAI-V1协议"""
    
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    
    _model_api_instance:ModelAPIBase = None
    """各种模型API的执行实例"""
    prompt_template: PromptTemplateBase = None
    """提示词模板"""

    @property
    def model_name(self):
        """运行模型名称，如：gpt-3.5-turbo、vicuna-13b、text-embedding-ada-002等"""
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        model_config = ManagementModels.get_model(self.model_type, self.generate_mode(), value)
        if not model_config:
            raise Exception(f'没有找到有效的模型，模型名称无效: [{value}]或没有指定默认的模型。')
        self._model_name = model_config['name']
        self.model_config = model_config
        self._model_api_instance = None
    
    @abstractmethod
    def generate_mode(self) -> Config.ModelGenerateModeType:
        '''模型运行的模式，如：chat、text、embedding、audio等'''
        pass
    
    def __init__(
            self, 
            model_type:str, 
            model_name:str = None,
            prompt_template:PromptTemplateBase = None,
            **model_kwargs: Any
            ):
        """
        创建大模型处理的实例，初始化一些环境变量，设置模型的具体名称等
        """
        self.prompt_template = prompt_template
        self.model_kwargs = model_kwargs

        self.model_type_config = ManagementModels.get_model_type(model_type)
        if not self.model_type_config:
            raise Exception(f'请指定有效的模型类型：[{model_type}]')
        self.model_type = self.model_type_config['model_type']
        self.api_protocol = self.model_type_config["api_protocol"]
        self.api_config = self.model_type_config["api_config"]

        # 会根据名字和alias查找对应的模型，同步设置self.model_config
        self.model_name = model_name
        
    def _get_model_api_instance(self) -> ModelAPIBase:
        '''
        根据api_protocol的配置创建模型API运行的实例: \
        api_protocol：module:class, 处理该模型类型的module和class, \
        module文件必须放置在models目录下，建议命名为：公司_api.py, \
        class在该文件中定义，建议命名为：公司_模型_api，必须继承自ModelAPIBase。
        '''
        if not self._model_api_instance:
            import importlib
            api_module, api_class = self.api_protocol.split(':')
            module = importlib.import_module("bida.models."+api_module)
            cls = getattr(module, api_class)
            self._model_api_instance = cls(self.api_config, self.model_config)
        return self._model_api_instance

    def get_main_prompt_from_template(self, prompt_template: PromptTemplateBase):
        '''从提示词模板中获取主提示词'''
        result = ""
        # 如果新传入的有提示词模板，就覆盖创建时的提示词模板
        prompt_template = prompt_template or self.prompt_template
        # 根据提示词模板和用户输入，拼装提示词
        if prompt_template:
            result = prompt_template.format(model=self)
        return result
    
    _log_trace_id: str = None
    _begintime: None

    def log(self, prefixstr, content):
        Util.log_info(f"[{self._log_trace_id}] {prefixstr} {content}")

    def log_begin(self, prefixstr):
        self._log_trace_id = str(uuid.uuid4())
        self._begintime = datetime.now()
        self.log(prefixstr, "begin......")

    def log_end(self, prefixstr):
        endtime = datetime.now()
        executiontime = endtime - self._begintime
        self.log(prefixstr, f"{self._begintime.strftime('%Y-%m-%d %H:%M:%S')} ~ {endtime.strftime('%Y-%m-%d %H:%M:%S')} ({executiontime.total_seconds():.2f} seconds)")
            
        self.log(prefixstr, "end.")