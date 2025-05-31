from tinygrad import Tensor, dtypes


def llama_logits_sample(
    logits: Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    if temperature < 1e-6:
        return logits.argmax().unsqueeze(0)
    logits = logits.flatten()

    logits = (logits != logits).where(-float("inf"), logits)

    t = (logits / temperature).softmax()

    counter, counter2 = (
        Tensor.arange(t.numel(), device=logits.device).contiguous(),
        Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous(),
    )

    if top_k > 0:
        output, output_indices = (
            Tensor.zeros(top_k, device=logits.device).contiguous(),
            Tensor.zeros(top_k, device=logits.device, dtype=dtypes.int32).contiguous(),
        )
        for i in range(top_k):
            t_argmax = (
                t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1
            ).cast(dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, top_k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(
                ((i, top_k - i - 1),)
            )
            t = (counter == t_argmax).where(0, t)

        output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
        output = (output_cumsum >= (1 - top_p)) * output
        output_indices = (output_cumsum >= (1 - top_p)) * output_indices

        output_idx = output.multinomial()
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial()

    return output_token
