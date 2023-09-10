import json

"""The name MUST BE the SAME as the FUNCTION NAME in the JSON FILE, CASE SENSITIVE"""

def search_google(query):
    """If the required information cannot be found, use Google search online"""
    """当前为演示示例"""
    
    query_info = {
        "query": query,
        "result": "No relevant information found.",
    }
    return json.dumps(query_info)

def search_baidu(query):
    """I如果查不到答案需要联网搜索且google无法使用时，使用百度搜索"""
    """当前为演示示例"""
    
    query_info = {
        "query": query,
        "result": "答案是：我也不知道，哈哈哈，还是我更懂中文吧~~~",
    }
    return json.dumps(query_info)

