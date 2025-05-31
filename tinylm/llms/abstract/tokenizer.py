from dataclasses import dataclass
from typing import List


@dataclass
class AbstractTokenizerWithInputs:
    input_ids: List[int]


class AbstractTokenizerForInference:
    pad_token_id: int
    eos_token_id: int

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, *args, **kwds) -> AbstractTokenizerWithInputs:
        raise NotImplementedError("Subclasses must implement this method.")

    def decode(self, *args, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
