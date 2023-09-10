from abc import ABC, abstractmethod
import uuid
import json

from bida.core.conversation import Conversation, Message, MessageCategory, MessageStatus

class BufferControllerBase(ABC):
    '''
    会话记录的缓存管理基类
    '''
    @abstractmethod
    def contraction(self, conversation: Conversation):
        '''根据buffer控制类的方式收缩会话对象'''

class MessagePairBufferController(BufferControllerBase):
    '''
    消息对（一问一答是一对）缓存模式：只保留最新的[n]对消息
    '''
    _pair_count: int

    def __init__(self, pair_count:int = 5) -> None:
        '''只保留最新的5对消息，System类的消息不计算在内'''
        super().__init__()
        self._pair_count = pair_count

    def contraction(self, conversation: Conversation):
        history = conversation.history_for_llm
        log_message_id = str(uuid.uuid4())
        id_list = []
        # 倒着跳过指定对数的消息后开始遍历处理
        if history and history[:-(self._pair_count*2)]:
            msg = history[:-(self._pair_count*2)][-1]
            if msg:
                # 如果是成对的对话，就全部归档，否则忽略这个消息，处理前面的内容
                if msg.role == 'assistant':     
                    msg.status = MessageStatus.archived
                    msg.link_message_id = log_message_id
                    id_list.insert(0, msg.id)
                for msg in reversed(history[:-(self._pair_count*2 + 1)]):
                    if msg.role == 'system':
                        break
                    else:
                        msg.status = MessageStatus.archived
                        msg.link_message_id = log_message_id
                        id_list.insert(0, msg.id)
        if id_list:
            conversation.append_message(
                id=log_message_id,
                role='buffer_contraction',
                content=json.dumps(
                    {
                        'controller': self.__class__.__name__,
                        'ids': id_list
                    }
                    ),
                display_content='',
                status=MessageStatus.completed,
                category=MessageCategory.actionlog,
                )
