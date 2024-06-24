import torch
from torch import nn


class Embed(nn.Module):
    def __init__(self, observation_size, action_size, batch_size, hidden_dims: int=[1024, 1024], output_dims=[32]):
        super(Embed, self).__init__()
        self._encoder = nn.Sequential()
        input_size = (observation_size + action_size) * batch_size
        hidden_dims = [input_size] + hidden_dims
        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:
                self._encoder.append(nn.Linear(hidden_dims[i - 1], hidden_dim))
                self._encoder.append(nn.ReLU())
        self._heads = nn.ModuleList()
        for output_dim in output_dims:
            head = nn.Linear(hidden_dim, output_dim)
            self._heads.append(head)
    def forward(self, x):
        x = self._encoder(x)
        xs = []
        for head in self._heads:
            xs.append(head(x))
        return x, xs
