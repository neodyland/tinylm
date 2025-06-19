from ..abstract.optimizer import AbstractOptimizer
from tinygrad.nn.optim import AdamW
from tinygrad import Tensor
from typing import List


class AdamWOptimizer(AbstractOptimizer):
    def __init__(
        self,
        model: List[Tensor],
        lr=1e-3,
        weight_decay=0.01,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    ):
        self.optimizer = AdamW(
            model, lr=lr, weight_decay=weight_decay, b1=b1, b2=b2, eps=eps
        )

    def step(self):
        return self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()
