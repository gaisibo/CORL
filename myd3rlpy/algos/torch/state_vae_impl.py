import copy
from typing import Optional, Sequence, List, cast, Union

import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api, to_cpu, to_cuda, get_state_dict, set_state_dict, map_location
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.base import ImplBase

from myd3rlpy.models.vaes import create_vae_factory, VAEFactory
from myd3rlpy.models.torch.vaes import VAE
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class StateVAEImpl(ImplBase):

    _learning_rate: float
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _factory: VAEFactory
    _use_gpu: Optional[Device]
    _vae: Optional[VAE]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        feature_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        factory: VAEFactory,
        replay_type: str,
        replay_lambda: float,
        gem_alpha: float,
        agem_alpha: float,
        ewc_rwalk_alpha: float,
        use_gpu: Optional[Device],
    ):
        self._observation_shape = observation_shape
        self._feature_size = feature_size
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._factory = factory
        self._replay_type = replay_type
        self._replay_lambda = replay_lambda
        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._use_gpu = use_gpu

        # initialized in build
        self._vae = None
        self._optim = None

    def build(self) -> None:
        # setup torch models
        self._build_vae()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()



    def to_gpu(self, device: Device = Device()) -> None:
        self._device = f"cuda:{device.get_id()}"
        to_cuda(self, self._device)

    def to_cpu(self) -> None:
        self._device = "cpu:0"
        to_cpu(self)

    def save_model(self, fname: str) -> None:
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

    def save_clone_data(self):
        assert self._vae is not None
        self._clone_vae = copy.deepcopy(self._vae)

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return 1
