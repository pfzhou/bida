## function接入手册


1. bida\functions\目录下建立一个“json”文件

参看[_json_template.json](_json_template.json)
```json
{
    "function_protocol": "py file name",
    "functions":
    [
        {
            "name": "function name",
            "description": "The detailed description of the function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "The detailed description of the param."
                    },
                    "param2": {
                        "type": "string",
                        "enum": ["type1", "type2"],
                        "description": "The detailed description of the param."
                    }
                },
                "required": ["param1"]
            }
        }
    ]
}
```
> **function_protocol**: 指定了对应函数的处理代码的文件名（不含.py）
>
> **functions**: 包含了本json中所有的函数定义

2. 编写具体函数实现方法的“py”文件
- 按照[function_protocol]中指定的py文件名创建文件
- 为每个函数实现代码，函数名、参数必须与json中的保持一致

3. 启动框架调试

可以在[examples]目录下新建一个jupyter notebook
```python
from bida import ChatLLM
from bida import ManagementFunctions

llm = ChatLLM(model_type='openai', model_name='3.5-0613')       # 只有0613及之后的模型支持function call

def my_stream_process_data(data):
    if isinstance(data, str):           
        print(data+'', end="", flush=True)
    else:
        print(f'函数调用中：{data}\n......')

message = '北京的天气？'
result = llm.chat(prompt=message, 
        stream_callback=my_stream_process_data,
        functions=ManagementFunctions.get_functions(),
        function_call="auto",  # auto is default, but we'll be explicit
        )
print(result)
```
