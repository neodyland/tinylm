class AbstractOptimizer:
    def step(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def zero_grad(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
