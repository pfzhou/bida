import requests
import json
import time

from bida import Util

class baidu_wenxin():
    '''
    百度文心一言的http调用封装类，支持大模型和Embedding模型的调用
    '''
    access_token_url = None
    model_baseurls = {}
    api_key = None
    secret_key = None

    POST_HEADERS = {'Content-Type': 'application/json'}

    _access_token: str = None
    '''根据key生成的access token， 有效期默认30天，本程序24小时刷新一次'''
    _access_token_cache_expiretion_time: float = None
    ''' access token缓存的过期时间'''

    @classmethod
    def get_access_token(cls, force_refresh: bool = False, token_cache_hours = 24):
        """
        使用 API Key，Secret Key 获取access_token，自动缓存24小时(默认)，\
        如果force_refresh=True，就重新获取并缓存。
        """
        if force_refresh or \
            not cls._access_token or \
            not cls._access_token_cache_expiretion_time or \
            cls._access_token_cache_expiretion_time < time.time():

            url = cls.access_token_url.format(cls.api_key, cls.secret_key)
            payload = json.dumps("")
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
                }
            
            response = requests.request("POST", url, headers=headers, data=payload)
            cls._access_token = response.json().get("access_token")
            cls._access_token_cache_expiretion_time = time.time() + token_cache_hours * 60 * 60      # 24 hours * 60 minutes * 60 seconds
        
        return cls._access_token
    

    @classmethod
    def call(
        cls, 
        model: str,
        prompt: str,
        history: list,
        temperature: float,                # 默认0.95，范围 (0, 1.0]，不能为0，ERNIE-Bot和turbo支持
        stream: bool,  
        *args, **kwargs   
    ):
        
        url = None
        messages = None

        if model is not None:
            url = baidu_wenxin.model_baseurls[model]
        if url is None:
            raise Exception("请指定有效的模型名称。")
        
        if not prompt or prompt.replace("\n", " ").strip() == '':
            raise Exception("请输入有效的问题。")
        prompt = prompt.replace("\n", " ").strip()
        if history is None:
            new_history = []
        else:
            new_history = history[:]
        new_history.append({"role": "user", "content": prompt})


        if messages is None:
            messages = {}
        messages["messages"] = new_history

        # 设定temperature(0-1]
        if temperature is not None:
            if temperature == 0:        # 不能为0，将默认值调整为0.01
                temperature = 0.01       
            messages['temperature'] =temperature
            
        if stream is not None:
            messages["stream"] = stream

        # 添加其他参数
        for key, value in kwargs.items():
            messages[key] = value
                
        payload = json.dumps(messages)
        try:
            access_token = baidu_wenxin.get_access_token()
            url = url + "?access_token={}".format(access_token)
            response = requests.request("POST", url, headers=baidu_wenxin.POST_HEADERS, data=payload, stream=stream)
            if stream:
                if response.status_code == 200:
                    return response.iter_lines()
                else:
                    raise Exception(f"调用模型出错，状态码：{response.status_code}，错误描述：{response.text}")
            else:
                return response.json()
        except Exception as e:
            Util.log_error(e)
            raise e
        
    @classmethod
    def embeddings_call(
        cls, 
        input,
        model: str,
        ):
        '''
        输入文本以获取embeddings。说明：\
        （1）文本数量不超过16 \
        （2）每个文本长度不超过 384个token \
        '''
        url = None

        if model is not None:
            url = baidu_wenxin.model_baseurls[model]
        if url is None:
            raise Exception("请指定有效的Embedding模型名称。")
        
        payload = json.dumps({"input": input})
        try:
            access_token = baidu_wenxin.get_access_token()
            url = url + "?access_token={}".format(access_token)
            response = requests.request("POST", url, headers=baidu_wenxin.POST_HEADERS, data=payload)
            return response.json()
        except Exception as e:
            Util.log_error(e)
            raise e

        