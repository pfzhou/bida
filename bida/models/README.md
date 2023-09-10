## 模型接入手册


### 1. bida\models\目录下建立一个“json”文件

参看[_json_template.json](_json_template.json)
```json
{    
    "model_type": "company name",
    "alias": "easy to remember name",
    "api_protocol": "py file name:class name", 
    "api_config":{   
        "api_key": {
            "default": "",
            "env":"your api key"
        } 
    },
    "chat_completions": [
        {
            "name": "model name",
            "alias": "easy to remember name",
            "max_tokens": 4096,
            "is_default": true,
            "description": "model description"
        }
    ],
    "completions": [
    ],
    "embeddings": [
    ]
}
```
> **api_protocol**: 指定了对应模型的处理代码位置结构为[py文件名]:[文件总的类]，注意使用“:”分割。
>
> **api_config**: 是模型自定义的配置节，优先级是: 环境变量（.env） > default value。如果不配置[env]的值，将不从环境变量中查找。
>
> **支持的模型类型**：[chat_completions], [completions], [embeddings], 在各自的配置节中配置具体的模型名称和信息，别名用于在代码中快速指定模型名称，[is_default]：标记当前模型是否是这个分类下默认运行的模型。

### 2. 编写具体模型实现方法的“py”文件
- 按照[api_protocol]中指定的py文件名创建文件。

模型支持那种能力，就继承对应的基类，然后实现相关的抽象方法。
**注意**：[ModelAPIBase]中的初始化函数：[init_Environment]不要忘记了，它的信息就来自于json文件中的[api_config]。

- 具体代码可以参考当前目录下各模型的API文件：openai_api.py， baidu_api.py等。

### 3. 启动框架调试

可以在[examples]目录下新建一个jupyter notebook
```python
from bida import ChatLLM

llm = ChatLLM(model_type='your model type name')
result = llm.chat("你好呀，请问你是谁？") 
print(result)
```
**注意**：如果模型使用[api key]，需要去.env文件中添加，要与json文件中的配置一致。

### 4. API复用

#### 4.1 OpenAI API接口复用
使用[FastChat](https://github.com/lm-sys/FastChat)等部署的开源模型，提供的Web API接口遵循[OpenAI-Compatible RESTful APIs](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)，可以直接复用OpenAI的定义。

假设你本地使用FastChat部署了一个Vicuna 13B的模型，API地址是："http://10.10.10.10:8000/v1"，新建一个“json”文件：
```json
{    
    "model_type": "vicuna",
    "alias": "vicuna, vicuña",
    "api_protocol": "openai_api:openai_api_v1",   
    "api_config":{   
        "api_base": {
            "default": "http://10.10.10.10:8000/v1",
            "env":""
        },
        "api_key": {
            "default": "EMPTY",
            "env":""
        },
        "organization":{
            "default": "",
            "env":""
        },
        "proxy": {
            "default": null,
            "env":""
        }
    },
    "chat_completions": [
        {
            "name": "vicuna-13b",
            "alias": "vicuna13b, vicuna",
            "max_tokens": 2048,
            "is_default": false,
            "description": "vicuna适配“/v1/chat/completions”的模型，13B V1.3版本"
        }
    ]
}
```
把这个文件放到bida安装目录里面的"models"目录里面（注意不是当前源代码的目录里面，可以用 "pip show bida"查看安装位置） 。

然后按上面第三步建立测试文件就可以使用了。

#### 4.2 百度或阿里云上开源模型的快速接入
百度和阿里云上托管了大量的第三方开源模型，百度的大部分模型可以直接复用baidu-thirdmodels.json文件：
1. bida安装目录里面的"models"目录里面找到并打开该文件
2. 从百度网站上复制模型名称和请求地址，在"model_baseurls"下面新增调用url
3. 从百度网站上复制模型信息，在"chat_completions"下面新增模型配置信息
4. 注意："model_baseurls"和"chat_completions"新增的模型名称必须一样，否则无法关联
5. 重启开发工具或应用，测试使用

阿里云的模型复用与百度云类似，可以复用aliyun.json，但要注意的是：阿里云很多模型返回内容不标准，还需要修改API代码，可以从aliyun_api.py继承一个类，仅调整对应的代码即可。MiniMaxPro就是对MiniMax继承后的修改，不需要重头写。