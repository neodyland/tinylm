from tinygrad import Tensor
import torch
import numpy as np


class TorchTinygradCloseTest:
    def __init__(self, atol=1e-5, rtol=1e-5):
        self.atol = atol
        self.rtol = rtol

    def assert_close(self, tg_tensor: Tensor, torch_tensor: torch.Tensor, msg=None):
        tg_np: np.ndarray = tg_tensor.numpy()
        try:
            if torch_tensor.is_complex():
                torch_real = torch_tensor.real.numpy()
                torch_img = torch_tensor.imag.numpy()
                np.testing.assert_allclose(
                    tg_np[:, 0], torch_real, atol=self.atol, rtol=self.rtol
                )
                np.testing.assert_allclose(
                    tg_np[:, 1], torch_img, atol=self.atol, rtol=self.rtol
                )
            else:
                np.testing.assert_allclose(
                    tg_np, torch_tensor.numpy(), atol=self.atol, rtol=self.rtol
                )
        except AssertionError as e:
            if msg:
                raise AssertionError(f"{msg}\n{str(e)}")
            else:
                raise e

    def tinygrad(self, data: Tensor | int | float) -> Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def torch(self, data: torch.Tensor | int | float) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def run(self, data: np.ndarray | int | float):
        if isinstance(data, np.ndarray):
            xtg = Tensor(data, requires_grad=False).clone()
            xpt = torch.tensor(data, dtype=torch.float32)
        else:
            xtg = data
            xpt = data
        tg_out = self.tinygrad(xtg)
        torch_out = self.torch(xpt)
        self.assert_close(
            tg_out, torch_out, msg=f"Results do not match for input: {data}"
        )
