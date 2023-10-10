import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 如果在py文件中，执行这一句
sys.path.append(ROOT_PATH)

from datetime import datetime
import streamlit as st

import bida
from bida import Config
from bida import ChatLLM
from bida import Conversation
from bida import ManagementModels
from bida import ManagementFunctions
from bida import PromptTemplate_Text
from bida import MessagePairBufferController

def get_model_type_list():
    result = []
    for key, value in ManagementModels._registered_models.items():
        if (modellist := value.get(Config.ModelGenerateModeType.ChatCompletions.value)) and len(modellist) > 0:
            result.append(key.lower())
    result = sorted(result)
    first_model_type = 'openai'
    if result and first_model_type in result:
        result.remove(first_model_type)
        result.insert(0, first_model_type)
    return tuple(result)

def get_model_name_list(model_type):
    models = ManagementModels.get_model_type(model_type)[Config.ModelGenerateModeType.ChatCompletions.value]
    names = tuple(item['name'] for item in models)
    return names

def format_chat_content(content, start_time=None, end_time=None, tokens=0):
    time = end_time if end_time else start_time
    time = time.strftime('%Y-%m-%d %H:%M:%S')
    if start_time and end_time and start_time != end_time:
        execution_time_seconds = end_time-start_time
        execution_time_seconds = f' ({execution_time_seconds.total_seconds():.2f} seconds)'
    else:
        execution_time_seconds = ''
    if tokens > 0:
        tokens = f',  {tokens} tokens'
    else:
        tokens=''
    result = f'{content}\n\n[ {time}{execution_time_seconds}{tokens} ]'
    return result

# 初始化页面
st.title('bida chat playground')
st.subheader(f"使用bida({bida.__version__})+streamlit整合众多LLM")


# 初始化参数
model_type_options=get_model_type_list()

if 'model_type' not in st.session_state:
    st.session_state.model_type = model_type_options[0]

# 从url参数加载缓存
session_params = 'session_params'
if session_params not in st.session_state:
    params = st.experimental_get_query_params()
    st.session_state[session_params] = params
    if params.get('conversation_id'):
        conversation_id = params['conversation_id'][0]
        _conversation = Conversation('')
        _conversation.load_from_db(conversation_id)
        st.session_state.model_type = _conversation.conversation_model_type
        session_conversation = st.session_state.model_type + "_conversation"
        if session_conversation not in st.session_state:
            st.session_state[session_conversation] = _conversation
    

def set_model_type_selectbox_state():
    st.session_state.model_type = st.session_state.model_type_selectbox
model_type = st.sidebar.selectbox(
    '**请选择LLM所属公司**:',
    options=model_type_options,
    key="model_type_selectbox", 
    on_change=set_model_type_selectbox_state,
    index=model_type_options.index(st.session_state.model_type),
    )

model_name = st.sidebar.selectbox(
    '**请选择LLM模型**:',
    options=get_model_name_list(model_type),
    )

temperature_value = st.sidebar.slider(
    label= '**Temperature**:',
    min_value=0.0, 
    max_value=2.0, 
    value=1.0, 
    step=0.1,
    format='%f',
    )
max_tokens_value = st.sidebar.slider(
    label= '**Max_tokens_value**:',
    min_value=1, 
    max_value=4096, 
    value=1024, 
    step=1,
    )

system_prompt = st.sidebar.text_area("**System Prompt**:", height=300, max_chars=1024)


# 判断有无缓存聊天记录
session_conversation = model_type + "_conversation"
if session_conversation not in st.session_state:
    st.session_state[session_conversation] = None

# 显示聊天标题
if st.session_state[session_conversation]:
    title= st.session_state[session_conversation].conversation_title
    conversation_id = st.session_state[session_conversation].conversation_id
    st.markdown(f"**[{title}](conversation?conversation_id={conversation_id})**")

# 每次都刷新显示聊天记录
if st.session_state[session_conversation]:
    for message in st.session_state[session_conversation].history_for_visible:
        with st.chat_message(message.role):
            st.markdown(format_chat_content(message.display_content, message.start_time, message.end_time, message.tokens))

llm = None
if prompt := st.chat_input("Send a message"):
    pt = None
    if system_prompt.replace("\n", " ").strip():
        pt = PromptTemplate_Text(system_prompt.replace("\n", " ").strip())

    llm = ChatLLM(
        model_type=model_type,
        model_name=model_name, 
        prompt_template=pt,
        temperature=temperature_value, 
        max_tokens=max_tokens_value,
        auto_save_conversation=True,
        buffer_controller=MessagePairBufferController(pair_count=10)
        )
    if st.session_state[session_conversation] and len(st.session_state[session_conversation]) > 0:
        # 从缓存中加载历史聊天记录
        llm.conversation = st.session_state[session_conversation]
        
        if model_type.lower() == 'openai':
            # 如果system_prompt有修改，就更新内容
            message = llm.conversation[0]
            if message.role == 'system' and message.content != system_prompt.replace("\n", " ").strip():
                message.content = system_prompt.replace("\n", " ").strip()
                message.display_content = message.content
                message.tokens=llm._get_model_api_instance().get_text_token_count(message.content) 
    
    with st.chat_message("user"):
        st.markdown(format_chat_content(prompt, datetime.now()))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_placeholder.markdown('▌')
        for response in llm.achat(
            prompt,
            # functions=ManagementFunctions.get_functions(),
            # function_call='auto',
            ):
            full_response = response
            message_placeholder.markdown(full_response + "▌")
        message = llm.conversation[-1]
        message_placeholder.markdown(format_chat_content(message.display_content, message.start_time, message.end_time, message.tokens))
    # 把新的聊天记录放到缓存中
    st.session_state[session_conversation] = llm.conversation

