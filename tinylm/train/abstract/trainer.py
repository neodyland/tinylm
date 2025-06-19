from tinygrad import Tensor, TinyJit
from typing import List, Callable
from .optimizer import AbstractOptimizer


class AbstractTrainerCallback:
    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_step_end(self, step: int):
        pass

    def on_accumulate_step_end(self, step: int, loss: Tensor):
        pass


class StackedTrainerCallback(AbstractTrainerCallback):
    def __init__(self, callbacks: List[AbstractTrainerCallback]):
        callbacks = []

        def add_callback(cb: AbstractTrainerCallback):
            callbacks.append(cb)

        def iter_callbacks(fn: Callable[[AbstractTrainerCallback], None]):
            for cb in callbacks:
                fn(cb)

        super().__setattr__("add_callback", add_callback)
        super().__setattr__("iter_callbacks", iter_callbacks)

    def __getattribute__(self, name):
        if name == "iter_callbacks":
            return super().__getattribute__(name)
        elif name == "add_callback":
            return super().__getattribute__(name)
        if not hasattr(super(), name):
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

        def wrapper(*args, **kwargs):
            def call_callback(cb: AbstractTrainerCallback):
                method = getattr(cb, name)
                if callable(method):
                    return method(*args, **kwargs)

            self.iter_callbacks(call_callback)

        return wrapper


class AbstractTrainer[B]:
    def __init__(self, optimizer: AbstractOptimizer, accumulate_steps: int = 1):
        self.optimizer = optimizer
        self.accumulate_steps = accumulate_steps
        self.callbacks = StackedTrainerCallback([])

    def add_callback(self, cb: AbstractTrainerCallback):
        self.callbacks.add_callback(cb)

    def next_batch(self) -> B:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def forward(self, batch: B) -> Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @TinyJit
    def step(self, step: int):
        self.optimizer.zero_grad()
        for _ in range(self.accumulate_steps):
            b = self.next_batch()
            loss = self.forward(b)
            loss.backward()
            self.callbacks.on_accumulate_step_end(step, loss)
        self.optimizer.step()

    def train(self, steps: int):
        self.callbacks.on_train_start()
        for i in range(steps):
            self.step(i)
            self.callbacks.on_step_end(i)
        self.callbacks.on_train_end()
