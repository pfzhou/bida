from enum import Enum
import logging

class ModelGenerateModeType(Enum):
    ChatCompletions = 'chat_completions'
    Completions = 'completions'
    Embeddings = 'embeddings'

########################## debug & log #################################
bida_debug = False
"""调试开关，输出详细的运行信息便于调试"""

log_file_level = logging.INFO
"""设置log文件默认记录级别"""

log_file_path = "log/"
"""生成log文件默认路径"""

db_file_path = "db/"
"""存放数据库db文件的默认路径"""

db_file_name = 'bida.db'
"""数据库db文件的默认名称"""