import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 如果在py文件中，执行这一句
sys.path.append(ROOT_PATH)

from datetime import datetime
import json
import streamlit as st

from bida import Conversation


_conversation_id = ''
# 从url参数加载缓存
params = st.experimental_get_query_params()
if params.get('conversation_id'):
    _conversation_id = params['conversation_id'][0]

if _conversation_id:
    st.markdown(f"会话ID:  **{_conversation_id}**")
else:
    _conversation_id = st.text_input("会话ID：")

if _conversation_id != st.session_state.get('conversation_id',''):
    st.session_state['updated'] = True

if _conversation_id and st.session_state.get('updated'):
    st.session_state.conversation_id = _conversation_id
    st.session_state['updated'] = False
    _conversation = Conversation('')
    _conversation.load_from_db(_conversation_id)
    st.session_state.conversation = _conversation

if st.session_state.get('conversation'):
    st.markdown(f"""会话标题:  **{st.session_state.conversation.conversation_title}**
                
会话模型:  **{st.session_state.conversation.conversation_model_type}**

最后保存时间:  **{st.session_state.conversation.persist_time.strftime('%Y-%m-%d %H:%M:%S')}**
""")

    col1, col2 = st.columns(2)
    with col1:
        st.radio(
            "**请选择展示内容：**",
            ('LLM可见', '用户可见', '全部数据'),
            horizontal=True,
            key='radio_list')

    with col2:
        st.radio(
            "**请选择显示模式：**",
            ('全部展开', '全部折叠'),
            horizontal=True,
            key='radio_expand')

    def ShowJson(json_obj, expand):
        expand = True if expand == '全部展开' else False
        st.markdown(f"**message: {len(json_obj)}个**")
        st.json(json.dumps(json_obj, default=Conversation.custom_encoder, ensure_ascii=False, indent=4), expanded=expand)

    if st.session_state.radio_list == 'LLM可见':
        ShowJson(st.session_state.conversation.history_for_llm, st.session_state.radio_expand)
    elif st.session_state.radio_list == '用户可见':
        ShowJson(st.session_state.conversation.history_for_visible, st.session_state.radio_expand)
    else:
        ShowJson(st.session_state.conversation, st.session_state.radio_expand)
