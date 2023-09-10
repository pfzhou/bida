import json

from bida import Config

class ManagementModels:
    '''注册并管理Models'''
    
    _registered_models = {}  # 用于存储已注册model的字典对象

    @classmethod
    def register(cls, model_json_file_path: str):
        '''根据指定的模型配置json文件，加载模型配置'''
        
        def register_error(error_msg):
            error_template = f"模型类型[{model_type}]注册失败，错误原因：{error_msg}\n"
            raise Exception(error_template)
        
        # 从文件中加载模型的json配置
        with open(model_json_file_path, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)
        
        model_type = jsondata["model_type"].strip()
        if not model_type:
            register_error("请输入有效的模型类型名称")
        if cls.get_model_type(model_type=model_type):
            register_error("该类型已经注册，请注意，alias也不能相同。")
        
        # 添加模型配置到全局模型列表中
        cls._registered_models[model_type] = jsondata
        
    @classmethod
    def get_model_type(cls, model_type):
        """
        根据模型类型获取相关配置，类型名字不区分大小写，支持alias匹配。
        """
        model_type = model_type.lower()
        model = cls._registered_models.get(model_type)
        if not model:
            for key, value in cls._registered_models.items():
                alias = value.get("alias")
                if alias and ManagementModels.is_in_alias(model_type, alias):
                    model = cls._registered_models[key]
                    break
        return model

    @classmethod
    def get_model(cls, model_type, generate_mode: Config.ModelGenerateModeType, model_name=None):
        """
        根据模型类型、生成模式和名字获取模型，类型和名字不区分大小写，都支持alias匹配，/
        如果不输入模型名字，就取当前生成模式下默认的模型。
        """
        mt = cls.get_model_type(model_type)

        result = None

        # model_type有注册
        if mt:      
            # 生成模式有注册                    
            if generate_mode:   
                # 生成模式下注册的有模型        
                if (modellist := mt.get(generate_mode.value)) and len(modellist) > 0:
                    # 遍历查找名字或alias匹配
                    if model_name and model_name.strip():
                        # 如果模型名字不为空，就遍历查找名字或alias匹配的
                        model_name = model_name.strip()
                        for item in modellist:
                            if model_name == item["name"].strip() or ManagementModels.is_in_alias(model_name, item["alias"]):
                                result = item
                                break
                    else: # 如果模型名字输入为None或空字符串  
                        # 如果只注册了一个模型，就默认返回
                        if len(modellist) == 1:
                            result = modellist[0]
                        else:
                            # 如果注册了多个，就查找那个是default，如果没有就返回None
                            for item in modellist:
                                if item['is_default']:
                                    result = item
                                    break
            return result
    
    @staticmethod    
    def is_in_alias(name: str, alias: str) -> bool:
        if not alias:
            return False
        # 使用逗号分隔别名，兼容中文逗号
        aliases = alias.replace("，", ",").split(",")
        # 删除每个别名两边的空白字符并转换为小写
        aliases = [a.strip().lower() for a in aliases]
        # 检查 name（转换为小写后）是否在别名列表中
        return name.lower() in aliases
