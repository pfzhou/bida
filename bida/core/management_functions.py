import json
import os

from bida import Config

class ManagementFunctions:
    '''注册并管理functions'''
    
    _registered_functions = []  
    """已注册functions的配置信息"""
    _available_function_names = {}
    """已注册functions的函数对象"""

    @classmethod
    def register(cls, func_json_file_path: str):
        '''根据指定的function配置json文件，加载配置'''
        
        def register_error(function_name, error_msg):
            error_template = f"{filename} 中的function name: [{function_name}]注册失败，错误原因：{error_msg}\n"
            raise Exception(error_template)
        
        filename = os.path.basename(func_json_file_path)
        # 从文件中加载json配置
        with open(func_json_file_path, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)
        
        # 加载function列表
        functions = jsondata["functions"]
        if len(functions) > 0:
            # 获取module
            function_module = jsondata['function_protocol']
            
            import importlib
            module = importlib.import_module("bida.functions."+function_module)

            for func in functions:
                name = func['name']
                if cls.get_function(name):
                    register_error(name, "该function name已经注册")
                else:
                    function = getattr(module, name)
                    cls._registered_functions.append(func)
                    cls._available_function_names[name] = function

    @classmethod
    def get_function(cls, function_name):
        """
        根据function_name获取相关配置，名字区分大小写
        """
        for func in cls._registered_functions:
            if func['name'] ==function_name:
                return func
        return None
        
    @classmethod
    def get_functions(cls, function_name_list = None):
        """
        获取所有注册成功的function详细信息,
        function_name_list默认为None，返回所有的已注册函数，
        可以指定函数名字，如果查找到就返回对应的函数。
        """
        result = cls._registered_functions
        if function_name_list:
            if isinstance(function_name_list, str):
                function_name_list = [function_name_list]
            result = [func for func in cls._registered_functions if any(func['name'] == name for name in function_name_list)]

        return result
    
    @classmethod
    def get_available_function_names(cls):
        """获取所有注册的function执行对象"""
        return cls._available_function_names
