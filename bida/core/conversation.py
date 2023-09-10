from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from datetime import datetime
from enum import Enum
import json
import uuid
import os
from pydantic import BaseModel, Field

import duckdb

from bida import Config
from bida import Util

class MessageCategory(Enum):
    '''会话消息分类'''
    llm = 'llm'                             # llm消息信息，用户及LLM产生的信息
    custom = 'custom'                       # 自定义信息，例如：后台对消息做summary后生成的信息
    data = "data"                           # 数据类消息，例如：通过API或函数查询返回的数据
    actionlog = 'actionlog'                 # log型信息，记录操作、summary、archive、delete等的log信息
    
class MessageStatus(Enum):
    '''会话消息状态'''
    processing = 'processing'               # 处理中：正在处理中的消息，中间状态，完成后必须被置为completed，否则就是无效的消息
    completed = 'completed'                 # 已完成：用户可见的，且做为chat history需要提交给LLM的
    archived = 'archived'                   # 已归档：用户可见的，但因过长后台合并做summary或直接归档的，不需要再次提交给LLM
    deleted = 'deleted'                     # 已删除：手工标记为已删除，不需要提交给LLM也不需要展示给用户
    
class Message(BaseModel):
    '''会话消息对象'''
    id: str                                 # message id
    role: str                               # system, user, assistant, function
    content: Optional[str]                  # content
    display_content: Optional[str]          # content for display to user 
    name: Optional[str]                     # function name
    function_call: Optional[str]            # function call content
    tokens: int                             # current completion's(request or response) token count 
    start_time: datetime                    # send or begin response time
    end_time: datetime                      # send or end response time
    category: MessageCategory               # category
    status: MessageStatus                   # status
    link_message_id: Optional[str]          # link message's id
    update_time: Optional[datetime]         # 初始化update_time字段

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__.update_time = datetime.now()

    def __setattr__(self, name, value):
        '''除update_time外，其他属性修改内容时自动更新update_time为当前时间'''
        if name != "update_time":
            object.__setattr__(self, 'update_time', datetime.now())
        object.__setattr__(self, name, value)


class Conversation(List[Message]):
    '''会话对象，提供会话的记录、数据库持久化和从数据库加载的能力'''

    conversation_id: str
    conversation_title: str
    conversation_model_type: str
    persist_time: Optional[datetime] = None

    _kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, model_type, **kwargs: Any):
        self._kwargs = kwargs
        self.conversation_id = str(uuid.uuid4())
        self.conversation_title = ''
        self.conversation_model_type = model_type
    
    history_for_llm_category_tuple = (MessageCategory.llm, MessageCategory.custom)
    history_for_llm_status_tuple = (MessageStatus.completed,) 
    @property
    def history_for_llm(self) -> List[Message]:
        '''
        提交给LLM的chat messages 集合：
        类型：正常交互和因需要定制生成的 history_for_llm_category_tuple
        状态：已完成的 history_for_llm_status_tuple
        '''
        filtered_messages = [msg for msg in self if msg.category in self.history_for_llm_category_tuple and msg.status in self.history_for_llm_status_tuple]
        return filtered_messages

    history_for_visible_category_tuple = (MessageCategory.llm, MessageCategory.data)
    history_for_visible_status_tuple = (MessageStatus.completed, MessageStatus.archived)
    @property
    def history_for_visible(self) -> List[Message]:
        '''
        显示给用户的chat messages 集合：
        类型：正常交互和交互中查询等产生的 history_for_visible_category_tuple
        状态： 已完成的和已归档的 history_for_visible_status_tuple
        '''
        filtered_messages = [msg for msg in self if msg.category in self.history_for_visible_category_tuple and msg.status in self.history_for_visible_status_tuple]
        return filtered_messages
    
    @property
    def history_messages(self) -> List:
        return Conversation.convert_history_to_llm_general_format(self.history_for_visible)

    def append_message(self, role, content, **kwargs: Any) -> Message:
        '''添加会话消息'''
        def setvalue(param_name, default_value):
            return kwargs[param_name] if param_name in kwargs else default_value
        msg = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            display_content=setvalue('display_content', content),
            tokens=setvalue('tokens', 0),
            name=setvalue('name', None),
            function_call=setvalue('function_call', None),
            start_time=setvalue('start_time', datetime.now()),
            end_time=setvalue('end_time', datetime.now()),
            category=setvalue('category', MessageCategory.llm),
            status=setvalue('status', MessageStatus.completed),
            link_message_id=setvalue('link_message_id', None),
            )
        self.append(msg)
        if self.conversation_title == '' and msg.role == 'user':
            self.conversation_title = msg.display_content[:20]
        return msg
    
    @staticmethod
    def custom_encoder(obj):
        if isinstance(obj, BaseModel):
            return json.loads(obj.json())
        
    def __repr__(self):
        return json.dumps(self.__dict__, default=Conversation.custom_encoder, ensure_ascii=False, indent=4)

    @property
    def __dict__(self):
        '''实现自定义的对象序列化'''
        return {
            'id': self.conversation_id,
            'model_type': self.conversation_model_type,
            'title': self.conversation_title,
            'persist_time': self.persist_time.isoformat() if self.persist_time else self.persist_time,
            'messages': list(self)
        }
    
    @staticmethod
    def convert_history_to_llm_general_format(messagelist: Conversation):
        '''转换会话消息列表为LLM通用的消息模式（openai v1标准）'''
        result = []
        for msg in messagelist:
            if msg.name:
                result.append({"role": msg.role, "content": msg.content, "name": msg.name})
            elif msg.function_call:
                result.append({"role": msg.role, "content": msg.content, "function_call": msg.function_call})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result
    
    _conversation_db_name = Config.db_file_name         # 数据库名称
    _conversation_table_name = 'conversation'           # 会话表名称
    _message_table_name = 'conversation_message'        # 会话消息表名称
    
    # 创建conversation表
    sql_create_conversation = f"""
CREATE TABLE IF NOT EXISTS {_conversation_table_name} (
    conversation_id VARCHAR PRIMARY KEY,
    conversation_title VARCHAR,
    conversation_model_type VARCHAR,
    persist_time TIMESTAMP
)
"""  
    # 查询conversation表
    sql_select_conversation = f"""
SELECT conversation_id, conversation_title, conversation_model_type, persist_time
FROM {_conversation_table_name}
WHERE conversation_id = ?
"""
    # 插入conversation表
    sql_insert_conversation = f"""
INSERT INTO {_conversation_table_name} (
    conversation_id, 
    conversation_title, 
    conversation_model_type, 
    persist_time)
VALUES (?, ?, ?, ?)
"""
    # 更新conversation表
    sql_update_conversation = f"""
UPDATE {_conversation_table_name}
SET conversation_title = ?, conversation_model_type = ?, persist_time = ?
WHERE conversation_id = ?
"""        
    
        # 创建message表
    sql_create_conversation_message = f"""
CREATE TABLE IF NOT EXISTS {_message_table_name} (
    id VARCHAR,
    role VARCHAR,
    content VARCHAR,
    display_content VARCHAR,
    name VARCHAR,
    function_call VARCHAR,
    tokens INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    category VARCHAR, 
    status VARCHAR, 
    link_message_id VARCHAR,
    update_time TIMESTAMP,
    conversation_id VARCHAR,
    PRIMARY KEY (id, update_time)
)
"""
    # 查询message表
    sql_select_message_with_conversation_id = f"""
SELECT 
    t1.id, 
    t1.role, 
    t1.content, 
    t1.display_content, 
    t1.name, 
    t1.function_call, 
    t1.tokens, 
    t1.start_time, 
    t1.end_time, 
    t1.category, 
    t1.status, 
    t1.link_message_id, 
    t1.update_time
FROM {_message_table_name} t1
JOIN (
    SELECT conversation_id, id, MAX(update_time) AS latest_update_time
    FROM {_message_table_name}
    WHERE conversation_id = ?
    GROUP BY conversation_id, id
) t2 ON t1.conversation_id = t2.conversation_id AND t1.id = t2.id AND t1.update_time = t2.latest_update_time
ORDER BY 
    CASE 
        WHEN t1.role = 'system' THEN 1
        ELSE 2
    END,
    t1.end_time ASC;
"""
    
    sql_select_message_with_conversation_message_id = f"""
SELECT 
    id, 
    role, 
    content, 
    display_content, 
    name, 
    function_call, 
    tokens, 
    start_time, 
    end_time, 
    category, 
    status, 
    link_message_id, 
    update_time
FROM {_message_table_name}
WHERE conversation_id = ? AND id = ?
ORDER BY update_time DESC
"""


    # 插入message表
    sql_insert_message = f"""
INSERT INTO {_message_table_name} (
    id, 
    role, 
    content, 
    display_content, 
    name,
    function_call, 
    tokens, 
    start_time, 
    end_time,
    category, 
    status, 
    link_message_id, 
    update_time,
    conversation_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

    def get_db_connection(self):
        if not os.path.exists(Config.db_file_path):
            os.makedirs(Config.db_file_path)
        file_path = os.path.join(Config.db_file_path, self._conversation_db_name)
        conn = duckdb.connect(file_path)
        return conn

    def init_db(self, conn):
        # 创建conversation表
        conn.execute(self.sql_create_conversation)
        # 创建message表
        conn.execute(self.sql_create_conversation_message)

    def persist(self):
        '''
        将会话对象完整的保存到数据库中，并根据_persist_time判断本次需要更新的内容，
        已保存的Message如果有更新，生成一个新的Message版本。
        '''
        def insert_message_to_db(conn, msg):
            conn.execute(self.sql_insert_message, (
                                                    msg.id,
                                                    msg.role,
                                                    msg.content, 
                                                    msg.display_content, 
                                                    msg.name,
                                                    msg.function_call, 
                                                    msg.tokens, 
                                                    msg.start_time,
                                                    msg.end_time,
                                                    msg.category.value, 
                                                    msg.status.value, 
                                                    msg.link_message_id, 
                                                    msg.update_time,
                                                    self.conversation_id
                                                )
                        )
            
        conn = self.get_db_connection()
        try:
            self.init_db(conn)
            persist_time = datetime.now()
            if self.persist_time:
                # 判断Conversation的几个属性有无更新，生成update语句
                # 判断Message中有无更新，生成insert语句
                
                conv = conn.execute(self.sql_select_conversation, (self.conversation_id, )).fetchall()
                if not conv or len(conv) > 1:
                    raise Exception(f'会话数据：{self.conversation_id}出错，需要数据修复')
                
                msg_list = []
                for msg in self:
                    if msg.update_time > self.persist_time:
                        msg_list.append(msg)
                
                if len(msg_list) > 0:
                    conn.begin()
                    try:
                        # 保存新增或更新过的Message
                        for msg in msg_list:
                            insert_message_to_db(conn, msg)
                        # 更新会话
                        conn.execute(self.sql_update_conversation, (self.conversation_title,
                                                                    self.conversation_model_type,
                                                                    persist_time,
                                                                    self.conversation_id))
                        conn.commit() # 提交事务
                        # 更新提交时间为最新时间
                        self.persist_time = persist_time
                    except Exception as e:
                        conn.rollback() # 如果出现错误，回滚事务
                        raise e
                else:
                    # 更新会话
                    conn.execute(self.sql_update_conversation, (self.conversation_title,
                                                                self.conversation_model_type,
                                                                persist_time,
                                                                self.conversation_id))
                    # 更新提交时间为最新时间
                    self.persist_time = persist_time
            else:
                # 初次更新，全量保存
                if len(self) > 0:
                    conn.begin()
                    try:
                        # 保存全部Message
                        for msg in self:
                            insert_message_to_db(conn, msg)
                        # 保存会话
                        conn.execute(self.sql_insert_conversation, (self.conversation_id,
                                                                    self.conversation_title,
                                                                    self.conversation_model_type,
                                                                    persist_time))
                        conn.commit() # 提交事务
                        # 更新提交时间为最新时间
                        self.persist_time = persist_time
                    except Exception as e:
                        conn.rollback() # 如果出现错误，回滚事务
                        raise e
                else:
                    # 保存会话
                    conn.execute(self.sql_insert_conversation, (self.conversation_id,
                                                                self.conversation_title,
                                                                self.conversation_model_type,
                                                                persist_time))
                    # 更新提交时间为最新时间
                    self.persist_time = persist_time
        except Exception as e:
            Util.log_error(e)
            raise e
        finally:
            conn.close()

    def load_from_db(self, conversation_id):
        '''
        根据指定的会话id从数据库中加载所有的会话内容
        '''
        conn = self.get_db_connection()
        try:     
            conv = conn.execute(self.sql_select_conversation, (conversation_id, )).fetchall()
            if not conv:
                raise Exception(f'未查到会话数据：{conversation_id}')
            elif len(conv) > 1:
                raise Exception(f'会话数据：{conversation_id}出错，需要数据修复')
            db_conversation_id = conv[0][0]
            db_conversation = Conversation('')
            db_messages = conn.execute(self.sql_select_message_with_conversation_id, 
                                    (db_conversation_id, )).fetchall()
            for msg in db_messages:
                mm = Message(
                    id=msg[0],
                    role=msg[1], 
                    content=msg[2], 
                    display_content=msg[3], 
                    name=msg[4], 
                    function_call=msg[5], 
                    tokens=msg[6], 
                    start_time=msg[7], 
                    end_time=msg[8], 
                    category=msg[9], 
                    status=msg[10], 
                    link_message_id=msg[11], 
                    )
                mm.update_time = msg[12]
                db_conversation.append(mm)
            self.conversation_id, self.conversation_title, self.conversation_model_type, self.persist_time = conv[0]
            self.clear()
            self.extend(db_conversation)    
        finally:
            conn.close()
