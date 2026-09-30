import torch


def topk_correct(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    """Number of samples whose label is in the top-k predictions, for each k."""
    max_k = min(max(ks), logits.shape[1])
    top = logits.topk(max_k, dim=1).indices
    hits = top == y.unsqueeze(1)
    return {k: int(hits[:, :min(k, max_k)].any(dim=1).sum()) for k in ks}
