import os
import copy
from copy import deepcopy
import sys
import time
import math
import random
from typing import Any, Dict, Optional, Sequence, List, Union, Callable, Tuple, Generator, Iterator, cast
import types
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, DataLoader
from torch.distributions.normal import Normal

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
from d3rlpy.torch_utility import TorchMiniBatch, _get_attributes
from d3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch, Transition
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.td3_plus_bc import TD3PlusBC
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    DYNAMICS_NOT_GIVEN_ERROR,
    ActionSpace,
)
from d3rlpy.base import LearnableBase
from d3rlpy.iterators import TransitionIterator
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer, dynamics_reward_prediction_error_scorer, dynamics_prediction_variance_scorer
from d3rlpy.iterators.random_iterator import RandomIterator
from d3rlpy.iterators.round_iterator import RoundIterator
from d3rlpy.logger import LOG, D3RLPyLogger
import gym

from online.utils import ReplayBuffer
from online.eval_policy import eval_policy

from myd3rlpy.algos.fs import FSBase
from myd3rlpy.algos.torch.state_vae_impl import StateVAEImpl as StateVAEImpl
from myd3rlpy.models.vaes import VAEFactory, create_vae_factory
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class FS(FSBase, TD3PlusBC):
    r"""Twin Delayed Deep Deterministic Policy Gradients algorithm.
    TD3 is an improved DDPG-based algorithm.
    Major differences from DDPG are as follows.
    * TD3 has twin Q functions to reduce overestimation bias at TD learning.
      The number of Q functions can be designated by `n_critics`.
    * TD3 adds noise to target value estimation to avoid overfitting with the
      deterministic policy.
    * TD3 updates the policy function after several Q function updates in order
      to reduce variance of action-value estimation. The interval of the policy
      function update can be designated by `update_actor_interval`.
    .. math::
        L(\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \min_j Q_{\theta_j'}(s_{t+1}, \pi_{\phi'}(s_{t+1}) +
            \epsilon) - Q_{\theta_i}(s_t, a_t))^2]
    .. math::
        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\min_i Q_{\theta_i}(s_t, \pi_\phi(s_t))]
    where :math:`\epsilon \sim clip (N(0, \sigma), -c, c)`
    References:
        * `Fujimoto et al., Addressing Function Approximation Error in
          Actor-Critic Methods. <https://arxiv.org/abs/1802.09477>`_
    Args:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.
    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    # _dynamics: Optional[ProbabilisticEnsembleDynamics]
    _rollout_interval: int
    _rollout_horizon: int
    _rollout_batch_size: int
    _use_gpu: Optional[Device]
    _embed: bool

    def __init__(
        self,
        critic_replay_type='orl',
        critic_replay_lambda=1,
        actor_replay_type='orl',
        actor_replay_lambda=1,
        vae_replay_type = 'bc',
        vae_replay_lambda = 1,
        conservative_threshold=0.1,
        gem_alpha: float = 1,
        agem_alpha: float = 1,
        ewc_rwalk_alpha: float = 0.5,
        damping: float = 0.1,
        epsilon: float = 0.1,
        impl_name = 'co',
        # n_train_dynamics = 1,
        retrain_topk = 4,
        log_prob_topk = 10,
        experience_type = 'random_transition',
        sample_type = 'retrain',

        critic_update_step = 0,

        clone_critic = False,
        clone_actor = False,
        merge = False,
        coldstart_step = 5000,

        fine_tuned_step = 1,

        embed = False,

        **kwargs: Any
    ):
        super(FSBase, self).__init__(**kwargs)
        self._critic_replay_type = critic_replay_type
        self._critic_replay_lambda = critic_replay_lambda
        self._actor_replay_type = actor_replay_type
        self._actor_replay_lambda = actor_replay_lambda
        self._vae_replay_type = vae_replay_type
        self._vae_replay_lambda = vae_replay_lambda

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon

        self._impl_name = impl_name
        # self._n_train_dynamics = n_train_dynamics
        self._retrain_topk = retrain_topk
        self._log_prob_topk = log_prob_topk
        self._experience_type = experience_type
        self._sample_type = sample_type

        self._conservative_threshold = conservative_threshold

        self._critic_update_step = critic_update_step

        self._clone_critic = clone_critic
        self._clone_actor = clone_actor
        self._coldstart_step = coldstart_step
        self._merge = merge
        self._fine_tuned_step = fine_tuned_step

        self._embed = embed

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        if self._impl_name == 'td3_plus_bc':
            from myd3rlpy.algos.torch.fs_td3_plus_bc_impl import FSTD3PlusBCImpl as FSImpl
        elif self._impl_name == 'td3n':
            from myd3rlpy.algos.torch.fs_td3n_impl import FSTD3NImpl as FSImpl
        else:
            print(self._impl_name)
            raise NotImplementedError
        self._impl = FSImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            vae_learning_rate=self._vae_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            vae_optim_factory=self._vae_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            vae_factory = self._vae_factory,
            feature_size = self._feature_size,
            q_func_factory=self._q_func_factory,
            critic_replay_type=self._critic_replay_type,
            critic_replay_lambda=self._critic_replay_lambda,
            actor_replay_type=self._actor_replay_type,
            actor_replay_lambda=self._actor_replay_lambda,
            vae_replay_type=self._vae_replay_type,
            vae_replay_lambda=self._vae_replay_lambda,
            # conservative_threshold=self._conservative_threshold,
            gamma=self._gamma,
            gem_alpha=self._gem_alpha,
            agem_alpha=self._agem_alpha,
            ewc_rwalk_alpha=self._ewc_rwalk_alpha,
            damping=self._damping,
            epsilon=self._epsilon,
            tau=self._tau,
            n_critics=self._n_critics,
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            fine_tuned_step = self._fine_tuned_step,
            embed = self._embed,
        )
        self._impl.build()

    def _update(self, batch_random: TransitionMiniBatch, batch_outer: TransitionMiniBatch, batch_inner: TransitionMiniBatch, online: bool, batch_num: int, total_step: int, coldstart_step: Optional[int] = None, score=False) -> Dict[int, float]:
        if self._embed:
            self._impl.embed(batch_random)
        loss1 = self._update_inner(batch_inner, online, batch_num, total_step, coldstart_step)
        print()
        loss2 = self._update_outer(batch_outer, online, batch_num, total_step, coldstart_step, score)
        loss1.update(loss2)
        return None, loss1

    def _update_outer(self, batch: TransitionMiniBatch, online: bool, batch_num: int, total_step: int, coldstart_step: Optional[int] = None, score=False) -> Dict[int, float]:
        print("update outer")
        if coldstart_step is None:
            coldstart_step = self._coldstart_step
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        
        critic_loss, value_loss, value, y, diff, replay_critic_loss = self._impl.outer_update_critic(batch)
        metrics.update({"outer_critic_loss": critic_loss})
        metrics.update({"outer_value": value})
        metrics.update({"outer_y": y})
        metrics.update({"outer_replay_critic_loss": replay_critic_loss})

        # if self._grad_step % self._update_actor_interval == 0:
        actor_loss, _, _, replay_actor_loss = self._impl.outer_update_actor(batch)
        metrics.update({"outer_actor_loss": actor_loss})
        metrics.update({"outer_replay_actor_loss": replay_actor_loss})
        self._impl.update_critic_target()
        self._impl.update_actor_target()

        del self._impl._critic_state_dict
        del self._impl._targ_critic_state_dict
        # if self._grad_step % self._update_actor_interval == 0:
        del self._impl._actor_state_dict
        del self._impl._targ_actor_state_dict

        return metrics

    def _update_inner(self, batch: TransitionMiniBatch, online: bool, batch_num: int, total_step: int, coldstart_step: Optional[int] = None) -> Dict[int, float]:
        print("update inner")
        if coldstart_step is None:
            coldstart_step = self._coldstart_step
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        
        # critic_loss, value_loss, value, y, diff = self._impl.inner_update_critic(batch)
        # metrics.update({"inner_critic_loss": critic_loss})
        # metrics.update({"inner_value": value})
        # metrics.update({"inner_y": y})

        # if self._grad_step % self._update_actor_interval == 0:
        actor_loss, _, _ = self._impl.inner_update_actor(batch)
        metrics.update({"inner_actor_loss": actor_loss})
        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, real_action_size, batch_size = 64,
    ) -> Optional[List[Transition]]:
        return None
