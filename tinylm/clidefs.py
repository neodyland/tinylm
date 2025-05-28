from enum import Enum
from tinygrad.tensor import DType, dtypes
from typing import Literal


class Model(str, Enum):
    UNSLOTH_Llama_3_2_1B_Instruct = "unsloth/Llama-3.2-1B-Instruct"
    UNSLOTH_Qwen3_0_6B = "unsloth/Qwen3-0.6B"


type ModelLiteral = Literal["unsloth/Llama-3.2-1B-Instruct", "unsloth/Qwen3-0.6B"]


class CliDType(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __str__(self):
        return self.value

    @property
    def dtype(self) -> DType:
        if self.value == "float32":
            return dtypes.float32
        elif self.value == "float16":
            return dtypes.float16
        elif self.value == "bfloat16":
            return dtypes.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.value}")
