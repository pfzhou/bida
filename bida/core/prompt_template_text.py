from typing import (
    Any,
)
from bida.core.prompt_template_base import PromptTemplateBase

class PromptTemplate_Text(PromptTemplateBase):
    '''
    基于Text的提示词模板
    '''
    main_prompt =  ""

    def __init__(
            self, 
        prompt = "",
        **kwargs: Any
        ):
        
        super().__init__(**kwargs)
        self.main_prompt = prompt
        