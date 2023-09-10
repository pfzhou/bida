import pytest

from bida import Embedding

def setup():
    print("")

def teardown():
    print("")
str_num_1 = "hello world"
str_num_3 = ["hello world", "hello china", " ", " \n \n\n", "hello AI"]

@pytest.mark.run(order=0)
@pytest.mark.parametrize("num",[1, 3, 30]) 
def test_openai_embedding(num):

    em = Embedding(model_type='openai')
    if num == 1:
        result = em.create(str_num_1)
    elif num == 3:
        result = em.create(str_num_3)
    else:
        result = em.create(str_num_3*10)
    
    assert len(result) == num
    assert len(result[0]) == 1536   # 维度

@pytest.mark.parametrize("num",[1, 3, 30]) 
def test_baidu_embedding(num):

    em = Embedding(model_type='baidu')
    if num == 1:
        result = em.create(str_num_1)
    elif num == 3:
        result = em.create(str_num_3)
    else:
        result = em.create(str_num_3*10)
    
    assert len(result) == num
    assert len(result[0]) == 384   # 维度

@pytest.mark.parametrize("num",[1, 3, 30]) 
def test_aliyun_embedding(num):

    em = Embedding(model_type='aliyun')
    if num == 1:
        result = em.create(str_num_1)
    elif num == 3:
        result = em.create(str_num_3)
    else:
        result = em.create(str_num_3*10)
    
    assert len(result) == num
    assert len(result[0]) == 1536   # 维度

@pytest.mark.xfail(True, reason="超过最大token")
def test_xfail_openai_embedding():

    em = Embedding(model_type='openai')
    # This model's maximum context length is 8191 tokens
    largestr = str_num_1 * 9000     # 18000 tokens
    
    result = em.create(largestr)
    
    assert len(result) == 1

@pytest.mark.xfail(True, reason="超过最大token")
def test_xfail_baidu_embedding():

    em = Embedding(model_type='baidu')
    # embeddings max tokens per batch size is 384
    largestr = str_num_1 * 300     # 600 tokens
    
    result = em.create(largestr)
    
    assert len(result) == 1

@pytest.mark.xfail(True, reason="超过最大token")
def test_xfail_aliyun_embedding():

    em = Embedding(model_type='aliyun')
    # Range of input length should be [1, 2048]
    largestr = str_num_1 * 2000     # 4000 tokens
    
    result = em.create(largestr)
    
    assert len(result) == 1