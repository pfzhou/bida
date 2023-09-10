from abc import ABC, abstractmethod

from bida.core.model_api_base import ModelAPIBase

class EmbeddingAPIBase(ModelAPIBase, ABC):
    """
    Embedding向量生成能力模型API的基类, 
    定义了接入模型API需要实现的抽象方法。
    """
    
    ################################################
    #  如果要实现embedding 向量生成的能力            #
    #  以下@abstractmethod 1个方法必须在子类中实现   #
    ################################################
    
    @abstractmethod
    def embeddingcompletion(self, 
                            inputText,
                            *args, **kwargs
                            ):
        '''
        各模型embedding模式时具体实现, 
        inputText：可以是str，也可以是[str]，支持对一个字符串或字符串列表进行embedding， 
        如果输入是字符串列表，返回是多个embedding对象。 

        请注意：如果是字符串列表，会自动删除strip后为空的str，可能会导致返回list的长度改变。
        '''
