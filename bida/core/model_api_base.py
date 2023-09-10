from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv, find_dotenv

from bida import Util

class ModelAPIBase(ABC):
    '''
    模型API的基类
    '''
    api_config: dict = None
    """api的配置信息"""
    model_config: dict = None
    """模型的配置信息"""
    model_name: str = None
    """模型的名称"""

    def __init__(
            self,
            api_config, 
            model_config
            ):
        self.api_config = api_config
        self.model_config = model_config
        self.model_name = model_config['name']

        # 初始化API参数
        _ = load_dotenv(find_dotenv()) # read local .env file
        self.init_Environment()

    ################################################
    # 以下 @abstractmethod 1个 方法必须在子类中实现  #
    ################################################

    @abstractmethod
    def init_Environment(self):
        """根据模型的配置，参数初始化"""
        

    def log_init(self, **kwargs):
        logstr = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        logstr = f"{[self.__class__.__name__]}初始化模型配置参数：{logstr}"
        Util.log_info(logstr)

    @staticmethod
    def get_from_env(config_name: str):
        """从env配置中获取对应的配置信息"""
        result = os.environ.get(config_name)
        return result
    
    @staticmethod
    def get_config_value(config):
        """从配置信息中获取默认配置和env配置的值，env配置优先级高"""
        default_value = config['default']
        env_value = ModelAPIBase.get_from_env(config['env'])
        result = env_value if env_value else default_value
        return result
