import torch
import torch.nn as nn


class MLP(nn.Module):
    """2-layer MLP that maps a 512-D embedding to a scalar quality score."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 1024, out_dim: int = 1) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.ac = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.ac(x)
        x = self.l2(x)
        return x


class eDifFIQA(nn.Module):
    """FR backbone + MLP head. Returns ``(features, quality)`` or ``quality``."""

    def __init__(self, backbone: nn.Module, head: nn.Module, return_feat: bool = False) -> None:
        super().__init__()
        self.base_model = backbone
        self.mlp = head
        self.return_feat = return_feat

    def forward(self, x: torch.Tensor):
        feat = self.base_model(x)
        pred = self.mlp(feat)
        return (feat, pred) if self.return_feat else pred
