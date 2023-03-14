import torch
from torch import nn
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction


class EnsembleEncoder(nn.Module, Encoder):
    def __init__(self, encoders):
        assert len(encoders) > 0
        super().__init__()
        self._encoders = torch.nn.ModuleList(encoders)
    def forward(self, x, reduce_type="mean"):
        hs = torch.stack([encoder(x) for encoder in self._encoders], dim=0)
        if reduce_type == "mean":
            hs = torch.mean(hs, dim=0)
        elif reduce_type == 'min':
            hs, _ = torch.min(hs, dim=0)
        return hs
    def get_feature_size(self):
        return self._encoders[0].get_feature_size()
    @property
    def observation_shape(self):
        return self._encoders[0].observation_shape
    @property
    def last_layer(self):
        raise NotImplementedError
    @property
    def action_size(self):
        return self._encoders[0].action_size

class EnsembleEncoderWithAction(EncoderWithAction, nn.Module):
    def __init__(self, encoders):
        assert len(encoders) > 0
        super().__init__()
        self._encoders = torch.nn.ModuleList(encoders)
    def forward(self, x, actions, reduce_type="mean"):
        hs = torch.stack([encoder(x, actions) for encoder in self._encoders], dim=0)
        if reduce_type == "mean":
            hs = torch.mean(hs, dim=0)
        elif reduce_type == 'min':
            hs = torch.min(hs, dim=0)
        return hs
    def get_feature_size(self):
        return self._encoders[0].get_feature_size()
    @property
    def observation_shape(self):
        return self._encoders[0].observation_shape
    @property
    def last_layer(self):
        raise NotImplementedError
    @property
    def action_size(self):
        return self._encoders[0].action_size
