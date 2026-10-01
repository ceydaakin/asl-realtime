"""Sequence classifiers over landmark frames: GRU and Conv1D baselines and a temporal transformer.

All models take (x, mask): x is (B, T, F), mask is (B, T) with True for real
frames. Real frames come first and padding only at the end.

Predictions must not depend on how much padding follows the real frames: a
live camera window is often not padded at all. So the hidden state is
re-masked after every layer (padding stays exactly zero, which a convolution
treats like its own edge padding), normalization is per-frame, the GRU reads
left to right so it finishes the real frames before any padding, and the
transformer never attends to padded frames.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

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


def sinusoidal_positions(length: int, dim: int, device=None) -> torch.Tensor:
    """(length, dim) fixed encoding, so any window length works without retraining."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    return torch.stack([(pos * freq).sin(), (pos * freq).cos()], dim=-1).reshape(length, dim)


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim), nn.Dropout(dropout))

    def forward(self, h: torch.Tensor, m: torch.Tensor, attend: torch.Tensor) -> torch.Tensor:
        b, t, _ = h.shape
        q, k, v = self.qkv(self.norm1(h)).reshape(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attend,
                                           dropout_p=self.dropout if self.training else 0.0)
        h = h + self.proj(a.transpose(1, 2).flatten(2))
        return (h + self.mlp(self.norm2(h))) * m


class TransformerClassifier(_Classifier):
    def __init__(self, n_features: int, hidden: int = 192, layers: int = 4, heads: int = 4, ff: int = 384,
                 dropout: float = 0.2, num_classes: int = NUM_CLASSES):
        super().__init__(n_features, hidden, hidden, dropout, num_classes)
        if hidden % heads or hidden % 2:
            raise ValueError(f"hidden={hidden} must be even and divisible by heads={heads}")
        self.hparams = {"hidden": hidden, "layers": layers, "heads": heads, "ff": ff,
                        "dropout": dropout, "num_classes": num_classes}
        self.blocks = nn.ModuleList(_AttentionBlock(hidden, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)

    def encode(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        attend = m.transpose(1, 2).bool().unsqueeze(1)  # (B, 1, 1, T): every frame looks at real frames only
        h = (h + sinusoidal_positions(h.shape[1], self.hparams["hidden"], h.device)) * m
        for block in self.blocks:
            h = block(h, m, attend)
        return self.norm(h)


MODELS = {"gru": GRUClassifier, "conv1d": Conv1DClassifier, "transformer": TransformerClassifier}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}, expected one of {sorted(MODELS)}")
    return MODELS[name](n_features=n_features, **kwargs)


class LandmarkClassifier(nn.Module):
    """Normalizer + sequence classifier: raw landmark x/y in, logits out. This is what gets exported."""

    def __init__(self, name: str, normalizer: nn.Module, net: nn.Module):
        super().__init__()
        self.name = name
        self.normalizer = normalizer
        self.net = net

    @property
    def hparams(self) -> dict:
        return {"model": self.name, "norm": self.normalizer.mode, **self.net.hparams}

    def forward(self, xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(xy, mask), mask)
