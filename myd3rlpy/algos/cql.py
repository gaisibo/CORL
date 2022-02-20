from typing import Any, Dict, Optional, Sequence

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.dqn import DoubleDQN
from d3rlpy.algos.torch.cql_impl import CQLImpl, DiscreteCQLImpl
from d3rlpy.algos import CQL
from myd3rlpy.algos.torch.co_impl_2 import COImpl


class MyCQL(CQL):

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = COImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=1e-4,
            alpha_learning_rate=1e-4,
            phi_learning_rate=1e-4,
            psi_learning_rate=1e-4,
            actor_optim_factory=AdamFactory(),
            critic_optim_factory=AdamFactory(),
            temp_optim_factory=AdamFactory(),
            alpha_optim_factory=AdamFactory(),
            phi_optim_factory=AdamFactory(),
            psi_optim_factory=AdamFactory(),
            actor_encoder_factory=check_encoder("default"),
            critic_encoder_factory=check_encoder("default"),
            q_func_factory=check_q_func("mean"),
            replay_critic_alpha=1,
            replay_actor_alpha=1,
            replay_type="orl",
            gamma=0.99,
            gem_gamma=1,
            agem_alpha=1,
            tau=0.005,
            n_critics=2,
            initial_alpha=1.0,
            initial_temperature=1.0,
            alpha_threshold=10.0,
            conservative_weight=5.0,
            n_action_samples=10,
            soft_q_backup=False,
            use_gpu=check_use_gpu(True),
            scaler=None,
            action_scaler=None,
            reward_scaler=None,
        )
        self._impl.build()
