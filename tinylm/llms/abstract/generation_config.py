from typing import List, Union, Optional
from pydantic import BaseModel


class AbstractGenerationConfig(BaseModel):
    eos_token_id: Optional[Union[List[int], int]] = None
    pad_token_id: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
