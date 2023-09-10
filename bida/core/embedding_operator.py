from datetime import datetime

from bida import Config
from bida import Util
from bida.core.model_operator_base import ModelOperatorBase

class Embedding(ModelOperatorBase):
    '''
    生成向量的模型操作类
    '''

    def generate_mode(self) -> Config.ModelGenerateModeType:
        '''模型运行的模式：embedding'''
        return Config.ModelGenerateModeType.Embeddings
    
    def create(self, inputText, *args, **kwargs):
        '''
        inputText：可以是str，也可以是[str]，支持对一个字符串或字符串列表进行embedding，
        如果输入是字符串列表，返回是多个embedding对象。
        请注意，如果是字符串列表，会自动删除strip后为空的str，可能会导致返回list的长度改变。
        '''
        try:
            # 预处理传入的字符串或字符串列表
            inputText = self.preproce_inputtext(inputText)
            # 获取模型实例
            instance = self._get_model_api_instance()

            # 模型调用前的log
            prefixstr = f"[{self.model_type}] embeddings:"
            self.log_begin(prefixstr)
            self.log(prefixstr, f"model_name={instance.model_name}")

            # 调用模型的执行方法
            result = instance.embeddingcompletion(inputText=inputText, *args, **kwargs)

            # 提取token数
            prompt_tokens = result['usage']["prompt_tokens"]
            total_tokens = result['usage']["total_tokens"]

            # 将所有embedding提取为列表
            result = result["data"]
            if len(result) > 1:
                sorted_embeddings = sorted(result, key=lambda e: e["index"])
                result = [em["embedding"] for em in sorted_embeddings]
            else:
                result = [result[0]["embedding"]]

            # 模型调用完成后的log
            self.log(prefixstr, f"成功转换 {len(result)}个{len(result[0])}维的embedding")
            self.log(prefixstr, f"Prompt Token={prompt_tokens}, Total Tokens={total_tokens}")
            self.log_end(prefixstr)
            
            return result
        except Exception as e:
            Util.log_error(e)
            raise e

    def preproce_inputtext(self, inputText):
        if inputText is not None:
            if isinstance(inputText, list):
                inputText = [s.replace("\n", " ").strip() for s in inputText if s.replace("\n", " ").strip() != ""]
                if not inputText:
                    raise Exception("请输入需要转换的内容。")
            elif isinstance(inputText, str):
                inputText = inputText.replace("\n", " ").strip()
                if not inputText:
                    raise Exception("请输入需要转换的内容。")
                inputText = [inputText]
            else:
                raise Exception("输入的内容不是字符串或字符串列表。")
        else:    
            raise Exception("请输入需要转换的内容。")
        return inputText
    