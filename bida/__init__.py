import bida.config as Config
import bida.util as Util
from bida.core.management_models import ManagementModels
from bida.core.management_functions import ManagementFunctions
from bida.core.llm_chat_operator import ChatLLM
from bida.core.llm_text_operator import TextLLM
from bida.core.embedding_operator import Embedding
from bida.core.conversation import Conversation, Message, MessageCategory, MessageStatus
from bida.core.conversation_buffer import BufferControllerBase, MessagePairBufferController
from bida.core.prompt_template_base import PromptTemplateBase
from bida.core.prompt_template_text import PromptTemplate_Text


__version__ = "0.9.4"

__all__ = [
    "Config",
    "Util",
    "ManagementModels",
    "ManagementFunctions",
    "ChatLLM",
    "TextLLM",
    "Embedding",
    "Conversation", 
    "Message", 
    "MessageCategory", 
    "MessageStatus",
    "BufferControllerBase",
    "MessagePairBufferController",
    "PromptTemplateBase",
    "PromptTemplate_Text",
]

def register_object_from_json_config(management, jsonpath, description):
    '''
    遍历/bida/models/目录下所有的json配置文件，注册到系统中。\
    如果文件为"_"开头，就忽略不注册。
    '''
    import os
    # 程序运行的根目录
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir_path = os.path.join(ROOT_PATH, jsonpath) 
    for filename in os.listdir(models_dir_path):
        # 查找所有文件名非'_'开始的json文件
        if filename.endswith('.json') and not filename.startswith('_'):
            full_path = os.path.join(models_dir_path, filename)
            # 注册json文件
            management.register(full_path)
            Util.log_info(f"suncess register {description}：{full_path}")

# 初始化所有支持的模型配置
register_object_from_json_config(ManagementModels, 'bida/models', "models config")
# 初始化所有支持的函数配置
register_object_from_json_config(ManagementFunctions, 'bida/functions', "function config")