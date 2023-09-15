import ssl
import json
import hmac
import hashlib
import base64
import datetime
import requests
import websocket
import queue
import threading
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time

from bida import Util

class xfyun_xinghuo():
    '''
    讯飞星火大模型的调用封装类，支持大模型和Embedding模型的调用。
    '''
    api_key = None
    secret_key = None
    app_id = None
    model_baseurls = {}
    model_domains = {}
    
    @classmethod
    def get_authorization_url(cls, model_url, isembedding=False):
        
        host = urlparse(model_url).netloc
        path = urlparse(model_url).path

        # 假使生成的date和下方使用的date = Fri, 05 May 2023 10:43:39 GMT
        cur_time = datetime.datetime.now()
        date = format_date_time(mktime(cur_time.timetuple()))
        
        signature_origin  = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        if isembedding:
            signature_origin += "POST " + path + " HTTP/1.1"
        else:
            signature_origin += "GET " + path + " HTTP/1.1"
        """上方拼接生成的signature_origin字符串如下
        host: spark-api.xf-yun.com
        date: Fri, 05 May 2023 10:43:39 GMT
        GET /v1.1/chat HTTP/1.1
        """

        signature_sha = hmac.new(
            xfyun_xinghuo.secret_key.encode('utf-8'), 
            signature_origin.encode('utf-8'), 
            digestmod=hashlib.sha256
            ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{xfyun_xinghuo.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
                "authorization": authorization,     # 上方鉴权生成的authorization
                "date": date,                       # 步骤1生成的date
                "host": host                        # 请求的主机名，根据具体接口替换
            }
        result = model_url + '?' + urlencode(v)
        return result
    
    @classmethod
    def call(
            cls, 
            model: str,
            messages: list,
            temperature: float,
            max_tokens: int,
            stream: bool,  
            *args, **kwargs   
        ):
        url =None
        domain = None
        if model is not None:
            url = xfyun_xinghuo.model_baseurls[model]
            domain = xfyun_xinghuo.model_domains[model]
        if url is None or domain is None:
            raise Exception("请指定有效的模型名称。")
        header = {
                    "app_id": xfyun_xinghuo.app_id,
                }
        parameter = {
                        "chat": {
                                    "domain": domain,
                                    "temperature": temperature,
                                    "max_tokens": max_tokens, 
                                }
                    }
        # 添加其他参数
        for key, value in kwargs.items():
            parameter["chat"][key] = value

        payload = {
                    "message": {
                                    "text": messages
                                }
                }
        
        # 拼成ws参数
        gen_params = {
            "header": header,
            "parameter": parameter,
            "payload": payload
        }
        try:
            url = xfyun_xinghuo.get_authorization_url(url)
            
            client = xf_websocket_client(url, gen_params, stream)
            if stream:
                result = client.call_stream()
            else:
                result = client.call()

            return result
        except Exception as e:
            Util.log_error(e)
            raise e
        
    @classmethod
    def embeddings_call(
            cls, 
            model: str,
            text,
            *args, **kwargs
        ):
        '''
        输入文本以获取embeddings。说明：不超过256个字符，超过会只截取前256个字符进行向量化
        '''
        url =None
        if model is not None:
            url = xfyun_xinghuo.model_baseurls[model]
        if url is None:
            raise Exception("请指定有效的模型名称。")
        gen_params = {
            "header": {
                        "app_id": xfyun_xinghuo.app_id,
                    },
            "payload": {
                        "text": text
                    }
        }
        try:
            url = xfyun_xinghuo.get_authorization_url(url, True)
            response = requests.post(url=url, json=gen_params)
            return response
        except Exception as e:
            Util.log_error(e)
            raise e

class xf_websocket_client:
    def __init__(self, url, gen_params, stream=False):
        self.url = url
        self.gen_params = gen_params
        self.stream = stream
        if stream:
            self.q = queue.Queue()
        self.received_messages = []

        # websocket.enableTrace(True)
        self.ws_app = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
            )
    
    def call(self):
        self.run()
        return self.received_messages
    
    def call_stream(self):
        threading.Thread(target=self.run).start()
        while True:
            message = self.q.get()
            if message is None:
                break
            else:
                yield [message]
        return self.received_messages
    
    def run(self):
        self.ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=60)
    
    def __iter__(self):
        return self

    def __next__(self):
        result = self.q.get()
        if result is None:
            raise StopIteration
        else:
            return result

    def on_open(self, ws):
        ws.send(json.dumps(self.gen_params))

    def on_close(self, ws, one, two):
        if self.stream:
            self.q.put(None)

    def on_message(self, ws, message):
        self.received_messages.append(message)
        if self.stream:
            self.q.put(message)

    def on_error(self, ws, error):
        error_msg = f"调用模型出错: {error.args[0]}"
        raise Exception(error_msg)
