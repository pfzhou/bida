# -*- coding: utf-8 -*-
# 运行环境python3
# 如果使用出错请注意sse版本是否正确, pip3 install sseclient-py==1.7.2
import time
import uuid
import base64
import hashlib
import hmac
import requests
import sseclient

class TencentHY():
    base_url:str = None
    app_id:int = None
    secret_id:str = None
    secret_key:str = None

    @classmethod
    def call(
            cls,
            messages:list = None,       #会话内容,  按对话时间序排列，长度最多为40;输入 content 总数最大支持 3000token。
            temperature:float = 0.0,    #默认1.0，取值区间为[0.0, 2.0]
            top_p:float = 1.0,          #默认1.0，取值区间为[0.0, 1.0],非必要不建议使用, 不合理的取值会影响效果.建议该参数和 temperature 只设置1个，不要同时更改
            stream: bool = False,
            *args, **kwargs             #可以传入query_id 跟踪和服务器之间的交互
            ):

        timestamp = int(time.time()) + 10000
        
        payload = {
            "app_id": TencentHY.app_id,
            "secret_id": TencentHY.secret_id,
            "timestamp": timestamp,
            "expired": timestamp + 24 * 60 * 60,
            "messages": messages,
            "query_id": "test_query_id_" + str(uuid.uuid4()),
            "temperature": temperature,
            "top_p": top_p,
            "stream": 1 if stream else 0,
        }

        # 添加其他参数
        for key, value in kwargs.items():
            payload[key] = value
        
        sign_params = TencentHY.gen_sign_params(payload)
        signature = TencentHY.gen_signature(sign_params)

        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        
        url = TencentHY.base_url
        resp = requests.post(url, headers=headers, json=payload, stream=stream)
        
        if stream:
            # 如果使用出错请注意sse版本是否正确, pip3 install sseclient-py==1.7.2
            client = sseclient.SSEClient(resp)
            return client
        else:
            data_js = resp.json()
            return data_js


    @classmethod
    def gen_signature(cls, param):
        sort_dict = sorted(param.keys())
        sign_str = cls.base_url.removeprefix("https://") + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(param[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(cls.secret_key.encode('utf-8'),
                        sign_str.encode('utf-8'), hashlib.sha1).digest()
        signature = base64.b64encode(hmacstr)
        signature = signature.decode('utf-8')
        return signature

    @staticmethod
    def gen_sign_params(data):
        params = dict()
        params['app_id'] = data["app_id"]
        params['secret_id'] = data['secret_id']
        params['query_id'] = data['query_id']
        # float类型签名使用%g方式，浮点数字(根据值的大小采用%e或%f)
        params['temperature'] = '%g' % data['temperature']
        params['top_p'] = '%g' % data['top_p']
        params['stream'] = data["stream"]
        # 数组按照json结构拼接字符串
        message_str = ','.join(
            ['{{"role":"{}","content":"{}"}}'.format(message["role"], message["content"]) for message in data["messages"]])
        message_str = '[{}]'.format(message_str)

        params['messages'] = message_str
        params['timestamp'] = str(data["timestamp"])
        params['expired'] = str(data["expired"])
        return params
