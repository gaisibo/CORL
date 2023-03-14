from typing import cast
import torch
import torch.nn.functional as F
from torch import nn
from myd3rlpy.models.torch.encoders import EnsembleEncoder
from d3rlpy.models.torch.v_functions import ValueFunction


class EnsembleValueFunction(ValueFunction):  # type: ignore
    _encoder: EnsembleEncoder
    _fc: nn.Linear

    def forward(self, x: torch.Tensor, reduce_type = "mean") -> torch.Tensor:
        h = self._encoder.forward(x, reduce_type=reduce_type)
        return cast(torch.Tensor, self._fc(h))

    def compute_error(
        self, observations: torch.Tensor, target: torch.Tensor, reduce_type = "mean"
    ) -> torch.Tensor:
        v_t = self.forward(observations, reduce_type=reduce_type)
        loss = F.mse_loss(v_t, target)
        return loss
