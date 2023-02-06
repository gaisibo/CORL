from copy import deepcopy
import inspect
import types
import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from d3rlpy.argument_utility import check_encoder
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.policies import squash_action
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.td3_plus_bc_impl import TD3PlusBCImpl
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from myd3rlpy.algos.torch.co_deterministic_impl import CODeterministicImpl
from myd3rlpy.algos.torch.st_impl import STImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class COTD3PlusBCImpl(STImpl, TD3PlusBCImpl):
    def __init__(
        self,
        conservative_threshold: float,
        critic_replay_type: str,
        critic_replay_lamdba: str,
        actor_replay_type: str,
        actor_replay_lamdba: str,
        gem_alpha: float,
        agem_alpha: float,
        ewc_rwalk_alpha: float,
        damping: float,
        epsilon: float,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self._critic_replay_type = critic_replay_type
        self._actor_replay_type = actor_replay_type
        self._conservative_threshold = conservative_threshold

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon
