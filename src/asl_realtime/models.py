"""Baseline sequence classifiers over landmark frames.

All models take (x, mask): x is (B, T, F), mask is (B, T) with True for real
frames. Real frames come first and padding only at the end.

Predictions must not depend on how much padding follows the real frames: a
live camera window is often not padded at all. So the hidden state is
re-masked after every layer (padding stays exactly zero, which a convolution
treats like its own edge padding), normalization is per-frame, and the GRU
reads left to right so it finishes the real frames before any padding.
"""

import torch
from torch import nn

from .landmarks import NUM_CLASSES


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class _Classifier(nn.Module):
    def __init__(self, n_features: int, hidden: int, pooled: int, dropout: float, num_classes: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(pooled, num_classes))

    def encode(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        h = self.encode(self.embed(x * m) * m, m)
        return self.head(masked_mean(h, mask))


class GRUClassifier(_Classifier):
    def __init__(self, n_features: int, hidden: int = 256, layers: int = 2,
                 dropout: float = 0.3, num_classes: int = NUM_CLASSES):
        super().__init__(n_features, hidden, hidden, dropout, num_classes)
        self.hparams = {"hidden": hidden, "layers": layers, "dropout": dropout, "num_classes": num_classes}
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)

    def encode(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.gru(h)[0] * m


class _ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pointwise = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:  # h: (B, T, C)
        y = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        return (h + self.pointwise(self.norm(y))) * m


class Conv1DClassifier(_Classifier):
    def __init__(self, n_features: int, hidden: int = 256, blocks: int = 6, kernel: int = 5,
                 dropout: float = 0.3, num_classes: int = NUM_CLASSES):
        super().__init__(n_features, hidden, hidden, dropout, num_classes)
        self.hparams = {"hidden": hidden, "blocks": blocks, "kernel": kernel,
                        "dropout": dropout, "num_classes": num_classes}
        self.blocks = nn.ModuleList(_ConvBlock(hidden, kernel, dropout) for _ in range(blocks))

    def encode(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h, m)
        return h


MODELS = {"gru": GRUClassifier, "conv1d": Conv1DClassifier}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}, expected one of {sorted(MODELS)}")
    return MODELS[name](n_features=n_features, **kwargs)
