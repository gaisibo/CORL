from typing import List
import torch
from torch import nn

from d3rlpy.models.torch.encoders import Encoder


class Phi(nn.Module):
    def __init__(self, encoders: List[Encoder], hidden_dim: int=1024, output_dim: int=32):
        super(Phi, self).__init__()
        self._encoders = encoders
        self._backbone = nn.Sequential(
            nn.Linear(encoders[0].get_feature_size(), hidden_dim),
            nn.ReLU(),
        )
        self._head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, action, task_id):
        out = None
        for encoder in self._encoders:
            if out is None:
                out = encoder(x, action, task_id)
            else:
                out += encoder(x, action, task_id)
        out /= len(self._encoders)
        x = out
        x = self._backbone(x)
        x = self._head(x)
        return x

class Psi(nn.Module):
    def __init__(self, encoder: Encoder, hidden_dim: int=1024, output_dim: int=32):
        super(Psi, self).__init__()
        self._encoder = encoder
        self._backbone = nn.Sequential(
            nn.Linear(encoder.get_feature_size(), hidden_dim),
            nn.ReLU(),
        )
        self._head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, task_id):
        x = self._encoder(x, task_id)
        x = self._backbone(x)
        x = self._head(x)
        return x
