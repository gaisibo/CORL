import torch
import torch.nn
from myd3rlpy.algos.torch.plug.plug import Plug


class PackNet(Plug):
    def build(self, networks):
        self._masks = nn.ModuleDict()
        self._softs = nn.ModuleDict()

    def change_task(self, task_id):
