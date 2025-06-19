from tinygrad import Tensor, dtypes
from ..abstract.trainer import AbstractTrainer, AbstractTrainerCallback
from ..abstract.optimizer import AbstractOptimizer
from ...llms.abstract.causal_lm import LlamaAbstractCausalLMForTraining
from dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)


@dataclass
class LmBatch:
    input_ids: Tensor


class LmBasicTrainerCallback(AbstractTrainerCallback):
    def __init__(self, total_steps: int):
        self.console = Console()
        self.total_steps = total_steps
        self.progress = Progress(
            TextColumn("[bold blue]Training"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.task_id = None

    def on_train_start(self):
        self.progress.start()
        self.task_id = self.progress.add_task("Steps", total=self.total_steps)

    def on_step_end(self, step: int):
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=step)

    def on_train_end(self):
        self.progress.stop()


class LmTrainer(AbstractTrainer[LmBatch]):
    def __init__(
        self,
        model: LlamaAbstractCausalLMForTraining,
        optimizer: AbstractOptimizer,
        accumulate_steps: int = 1,
    ):
        super().__init__(optimizer, accumulate_steps)
        self.model = model

    def next_batch(self):
        return LmBatch(
            input_ids=Tensor.randint(4, 257, low=0, high=15000, dtype=dtypes.int64)
        )

    def forward(self, batch):
        targets = batch.input_ids[:, 1:]
        input_ids = batch.input_ids[:, :-1]
        logits = self.model(input_ids)
        return logits.view(-1, logits.size(-1)).cross_entropy(
            targets.view(-1), reduction="sum"
        )
