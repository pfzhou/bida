import pytest
import time

from bida import ChatLLM
from bida import TextLLM


def setup():
    print("")

def teardown():
    print("")

def my_stream_process_data( data):  
    print(data, end="", flush=True)


# 每个openai的test方法调用前，等待n秒钟
@pytest.fixture(scope="function", autouse=True)
def sleepseconds(seconds = 0):
    print(f'wait {seconds} seconds...')
    time.sleep(seconds)

################################### openai ##################################
# 测试可以调用
@pytest.mark.run(order=0)
def test_openai_chat():
    llm = ChatLLM(model_type="openai", max_tokens=5)
    assert llm.chat("hello") is not None

# 测试chat stream模式
def test_openai_stream_chat():
    llm = ChatLLM(model_type="openai", max_tokens=5)
    result = llm.chat("hello", stream_callback=my_stream_process_data)
    assert result is not None

# 测试会话历史是否有效
def test_openai_chat_history():
    llm = ChatLLM(model_type="openai", max_tokens=50)
    assert llm.chat("你好，你可以用老周称呼我") is not None
    assert llm.chat("你怎么称呼我？").index("老周") >= 0
    
# 测试text调用
def test_openai_text():
    llm = TextLLM(model_type="openai", max_tokens=5)
    assert llm.completion("hello") is not None

# 测试text stream模式
def test_openai_stream_text():
    llm = TextLLM(model_type="openai", max_tokens=5)
    result = llm.completion("hello", stream_callback=my_stream_process_data)
    assert result is not None

#################################### baidu ##########################################

# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("baidu","ernie-bot"), ("baidu","ernie-bot-turbo")]) 
def test_baidu_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("baidu","ernie-bot"), ("baidu","ernie-bot-turbo")]) 
def test_baidu_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("baidu","ernie-bot"), ("baidu","ernie-bot-turbo")]) 
def test_baidu_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("你好，我的名字是老周，你可以用老周称呼我") is not None
    assert llm.chat("我的名字是？").index("老周") >= 0

#################################### aliyun ##########################################

# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("aliyun","qwen-plus-v1"), ("aliyun","qwen-v1"), ("aliyun","qwen-7b-chat-v1")]) 
def test_aliyun_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("aliyun","qwen-plus-v1"), ("aliyun","qwen-v1"), ("aliyun","qwen-7b-chat-v1")]) 
def test_aliyun_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("aliyun","qwen-plus-v1"), ("aliyun","qwen-v1"), ("aliyun","qwen-7b-chat-v1")]) 
def test_aliyun_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("你好，我的名字是老周，你可以用“老周”称呼我") is not None
    result = llm.chat("你怎么称呼我？")
    assert result.index("老周") >= 0


#################################### minimax ##########################################
# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("minimax","abab5"), ("minimax","abab5.5")]) 
def test_minimax_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("minimax","abab5"), ("minimax","abab5.5")]) 
def test_minimax_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("minimax","abab5"), ("minimax","abab5.5")]) 
def test_minimax_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("你好，我的名字是老周，你可以用“老周”称呼我") is not None
    result = llm.chat("我的名字是？")
    print(result)
    assert result.index("老周") >= 0


#################################### zhipuai chatglm ##########################################
# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("chatglm","pro"), ("chatglm","std"), ("chatglm","lite")]) 
def test_zhipuai_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("chatglm","pro"), ("chatglm","std"), ("chatglm","lite")]) 
def test_zhipuai_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=50,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("chatglm","pro"), ("chatglm","std"), ("chatglm","lite")]) 
def test_zhipuai_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("你好，我的名字是老周，你可以用“老周”称呼我") is not None
    result = llm.chat("我的名字是？")
    print(result)
    assert result.index("老周") >= 0


#################################### senstime ##########################################
# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("senstime","xl"), ("senstime","xs")]) 
def test_senstime_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("senstime","xl"), ("senstime","xs")]) 
def test_senstime_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=50,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("senstime","xl"), ("senstime","xs")]) 
def test_senstime_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("你好，我的名字是老周") is not None
    result = llm.chat("我的名字是？")
    print(result)
    assert result.index("老周") >= 0

#################################### baichuan ##########################################
# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("baichuan","53b")]) 
def test_baichuan_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=5)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("baichuan","53b")]) 
def test_baichuan_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=50,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("baichuan","53b")]) 
def test_baichuan_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("我的名字是:老周") is not None
    result = llm.chat("你可以怎么称呼我？")
    print(result)
    assert result.index("老周") >= 0


#################################### tencent ##########################################
# 测试直接调用
@pytest.mark.parametrize("model_type, model_name",[("tencent","hy")])
def test_tencent_models_chat(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=20)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试steam调用
@pytest.mark.parametrize("model_type, model_name",[("tencent","hy")])
def test_tencent_models_chat_stream(model_type, model_name):
    llm = ChatLLM(model_type=model_type, max_tokens=20,stream_callback=my_stream_process_data)
    llm.model_name = model_name
    assert llm.chat("hello") is not None

# 测试会话历史是否有效
@pytest.mark.parametrize("model_type, model_name",[("tencent","hy")])
def test_tencent_chat_history(model_type, model_name):
    llm = ChatLLM(model_type=model_type, model_name=model_name, max_tokens=20)
    assert llm.chat("你好，我的名字是小吴，你可以称呼我'吴老师'") is not None
    result = llm.chat("你可以怎么称呼我？")
    print(result)
    assert result.index("吴老师") >= 0