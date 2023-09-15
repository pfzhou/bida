import requests
import json
import time
import hashlib



class baichuan():
    '''
    百川模型的http调用封装类，仅支持LLM模型调用，无embedding模型。
    仅支持chat模式，暂不支持temperature等模型超参数设置功能。
    支持一次返回和流式返回两种模式
    '''
    api_url_stream = None
    api_url = None
    api_key = None
    secret_key = None 

    @classmethod
    def calculate_md5(cls, input_string):
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        encrypted = md5.hexdigest()
        return encrypted

    @classmethod
    def _get_requests_header(cls,json_data_str:str,request_id:str = None):
        time_stamp = int(time.time())
        signature = cls.calculate_md5(cls.secret_key + json_data_str + str(time_stamp))
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cls.api_key,
            # "X-BC-Request-Id": request_id,
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }
        return headers

    @classmethod
    def call(
            cls,
            model: str ,
            chatMessages:list = None,
            stream: bool = False,
            *args, **kwargs
            ):
        request_id = kwargs.get("request_id",None)
        
        if stream:
            url = cls.api_url_stream
        else:
            url = cls.api_url
        data = {
            "model": model,
            "messages": chatMessages
        }

        json_data = json.dumps(data)
        headers = cls._get_requests_header(json_data,request_id)

        try:
            response = requests.post(url, data=json_data, headers=headers)
            if response.status_code == 200 and stream == False:
                return response.json()

            elif response.status_code == 200 and stream == True:
                return response.iter_lines()
            else:
                raise Exception(f"调用模型出错，状态码：{response.status_code}，错误描述：{response.text}")
        except Exception as e:
            raise e
