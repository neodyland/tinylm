from .lm.optimizer import AdamWOptimizer
from .lm.trainer import LmTrainer, LmBasicTrainerCallback
from ..llms.qwen3.model import Qwen3ModelForCasualLM
from tinygrad import nn, Tensor, dtypes

Tensor.training = True

if __name__ == "__main__":
    model = Qwen3ModelForCasualLM(
        num_layers=21,
        dim=768,
        ffn_dim=2304,
        kv_heads=6,
        head_dim=96,
        vocab_size=2000,
        rope_theta=1000000,
        att_heads=12,
        ctx_len=2048,
    )
    sd = {}
    params = 0
    for name, param in nn.state.get_state_dict(model).items():
        sd[name] = param.cast(dtypes.bfloat16)
        params += param.numel()
    print(f"Total parameters: {params / 1e6:.2f}M")
    nn.state.load_state_dict(model, sd)
    optimizer = AdamWOptimizer(nn.state.get_parameters(model))
    trainer = LmTrainer(model, optimizer)
    trainer.add_callback(LmBasicTrainerCallback(100))
    trainer.train(100)