import torch

from asl_realtime.metrics import topk_correct


def test_topk_correct_counts_hits():
    logits = torch.tensor([
        [0.1, 0.9, 0.0],   # top1 = 1
        [0.5, 0.2, 0.3],   # top1 = 0, top2 = {0, 2}
        [0.2, 0.3, 0.5],   # top1 = 2
    ])
    y = torch.tensor([1, 2, 0])

    assert topk_correct(logits, y, ks=(1, 2)) == {1: 1, 2: 2}


def test_topk_correct_caps_k_at_num_classes():
    logits = torch.tensor([[0.1, 0.9]])
    y = torch.tensor([0])

    assert topk_correct(logits, y, ks=(5,)) == {5: 1}
