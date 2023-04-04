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
    check_use_gpu,
)
from d3rlpy.torch_utility import TorchMiniBatch, _get_attributes
from d3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch, Transition
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.iql import IQL
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

from myd3rlpy.algos.st import ST
from myd3rlpy.algos.torch.state_vae_impl import StateVAEImpl as StateVAEImpl
from myd3rlpy.models.vaes import VAEFactory, create_vae_factory
from myd3rlpy.models.encoders import EnsembelDefaultEncoderFactory
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class ST(ST, IQL):
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
    _vae_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _vae_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _value_encoder_factory: EncoderFactory
    _vae_encoder_factory: EncoderFactory
    _vae_factory: VAEFactory
    _feature_size: int
    # actor必须被重放，没用选择。
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
    _critic_replay_type: bool
    _critic_replay_lambda: float
    _actor_replay_type: bool
    _actor_replay_lambda: float
    _replay_model: bool
    _generate_step: int
    _select_time: int
    _model_noise: float

    _task_id: str
    _single_head: bool
    _merge: bool

    def __init__(
        self,
        critic_replay_type='bc',
        critic_replay_lambda=1,
        actor_replay_type='rl',
        actor_replay_lambda=1,
        vae_replay_type = 'bc',
        vae_replay_lambda = 1,
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
        match_prop_quantile = 0.5,
        match_epsilon = 0.1,
        random_sample_times = 10,

        critic_update_step = 100000,

        clone_critic = False,
        clone_actor = False,
        merge = False,
        coldstart_step = 5000,

        fine_tuned_step = 1,
        n_ensemble = 10,
        expectile_max = 0.7,
        expectile_min = 0.7,
        std_lambda = 0.01,
        std_bias = 0,

        use_vae = False,
        vae_learning_rate = 1e-3,
        vae_optim_factory = AdamFactory(),
        vae_factory = 'vector',
        feature_size = 256,

        **kwargs: Any
    ):
        super().__init__(
            **kwargs,
        )
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

        self._begin_grad_step = 0

        self._match_prop_quantile = match_prop_quantile
        self._match_epsilon = match_epsilon
        self._random_sample_times = random_sample_times

        self._critic_update_step = critic_update_step

        self._clone_critic = clone_critic
        self._clone_actor = clone_actor
        self._coldstart_step = coldstart_step
        self._merge = merge
        self._fine_tuned_step = fine_tuned_step
        self._n_ensemble = n_ensemble
        self._expectile_min = expectile_min
        self._expectile_max = expectile_max
        self._std_lambda = std_lambda
        self._std_bias = std_bias

        self._use_vae = use_vae
        self._vae_learning_rate = vae_learning_rate
        self._vae_optim_factory = vae_optim_factory
        if isinstance(vae_factory, str):
            self._vae_factory = create_vae_factory(vae_factory)
        else:
            self._vae_factory = vae_factory
        self._feature_size = feature_size

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        impl_dict = {
            'observation_shape':observation_shape,
            'action_size':action_size,
            'actor_learning_rate':self._actor_learning_rate,
            'critic_learning_rate':self._critic_learning_rate,
            'vae_learning_rate':self._vae_learning_rate,
            'actor_optim_factory':self._actor_optim_factory,
            'critic_optim_factory':self._critic_optim_factory,
            'vae_optim_factory':self._vae_optim_factory,
            'actor_encoder_factory':self._actor_encoder_factory,
            'critic_encoder_factory':self._critic_encoder_factory,
            'value_encoder_factory':self._value_encoder_factory,
            'vae_factory': self._vae_factory,
            'use_vae': self._use_vae,
            'feature_size': self._feature_size,
            'critic_replay_type':self._critic_replay_type,
            'critic_replay_lambda':self._critic_replay_lambda,
            'actor_replay_type':self._actor_replay_type,
            'actor_replay_lambda':self._actor_replay_lambda,
            'vae_replay_type':self._vae_replay_type,
            'vae_replay_lambda':self._vae_replay_lambda,
            'gamma':self._gamma,
            'gem_alpha':self._gem_alpha,
            'agem_alpha':self._agem_alpha,
            'ewc_rwalk_alpha':self._ewc_rwalk_alpha,
            'damping':self._damping,
            'epsilon':self._epsilon,
            'tau':self._tau,
            'n_critics':self._n_critics,
            'expectile': self._expectile,
            'weight_temp': self._weight_temp,
            'max_weight': self._max_weight,
            'use_gpu':self._use_gpu,
            'scaler':self._scaler,
            'action_scaler':self._action_scaler,
            'reward_scaler':self._reward_scaler,
            'fine_tuned_step': self._fine_tuned_step,
            'n_ensemble': self._n_ensemble,
        }
        if self._impl_name == 'iql':
            from myd3rlpy.algos.torch.st_iql_impl import STImpl as STImpl
        elif self._impl_name == 'iqln':
            from myd3rlpy.algos.torch.st_iqln_impl import STImpl as STImpl
            impl_dict["expectile_max"] = self._expectile_max
            impl_dict["expectile_min"] = self._expectile_min
            impl_dict["std_lambda"] = self._std_lambda
            impl_dict["std_bias"] = self._std_bias
        self._impl = STImpl(
            **impl_dict
        )
        self._impl.build()

    # 注意欧氏距离最近邻被塞到actions后面了。
    def _update(self, batch: TransitionMiniBatch, online: bool, batch_num: int, total_step: int, coldstart_step: Optional[int] = None, replay_batch: Optional[List[Tensor]]=None) -> Dict[int, float]:
        if coldstart_step is None:
            coldstart_step = self._coldstart_step
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        if not self._merge or total_step < coldstart_step:
            critic_loss, replay_critic_loss = self._impl.update_critic(batch, replay_batch, clone_critic=self._clone_critic, online=online)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})

            if (total_step > self._critic_update_step and total_step < coldstart_step) or self._impl._impl_id == 0:
                actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
                # actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
                metrics.update({"actor_loss": actor_loss})
                metrics.update({"replay_actor_loss": replay_actor_loss})

            if self._use_vae and not online:
                vae_loss, replay_vae_loss = self._impl.update_vae(batch, replay_batch)
                # actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
                metrics.update({"vae_loss": vae_loss})
                metrics.update({"replay_vae_loss": replay_vae_loss})

                critic_loss = self._impl.update_vae_critic(batch)
                metrics.update({"critic_loss": critic_loss})

            self._impl.update_critic_target()
        elif not online:
            self._merge_update(batch,replay_batch)

        return metrics

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, real_action_size, batch_size = 64,
    ) -> Optional[List[Transition]]:
        return None

    def select_replay(self, new_replay_dataset, old_replay_dataset, dataset_id, max_save_num, mix_type='vq_diff'):
        if mix_type == 'vq_diff':
            new_replay_dataloader = DataLoader(new_replay_dataset, batch_size = self._batch_size, shuffle=False)
            new_replay_diff_qs = []
            for replay_observations, replay_actions, _, _, replay_terminals, *_ in new_replay_dataloader:
                replay_observations = replay_observations.to(self._impl.device)
                replay_actions = replay_actions.to(self._impl.device)
                replay_qs = self._impl._q_func(replay_observations, replay_actions)
                if self._impl_name == 'iql':
                    replay_vs = self._impl._value_func(replay_observations)
                elif self._impl_name == 'iqln':
                    replay_vs = []
                    for v_func in self._impl._value_funcs:
                        replay_vs.append(v_func(replay_observations))
                    replay_vs = torch.mean(torch.stack(replay_vs, dim=0), dim=0)
                new_replay_diff_qs.append(replay_qs - replay_vs)
            new_replay_diff_qs = torch.cat(new_replay_diff_qs, dim=0)

            old_replay_dataloader = DataLoader(old_replay_dataset, batch_size = self._batch_size, shuffle=False)
            old_replay_diff_qs = []
            for replay_observations, replay_actions, _, _, replay_terminals, *_ in old_replay_dataloader:
                replay_observations = replay_observations.to(self._impl.device)
                replay_actions = replay_actions.to(self._impl.device)
                replay_qs = self._impl._q_func(replay_observations, replay_actions)
                if self._impl_name == 'iql':
                    replay_vs = self._impl._value_func(replay_observations)
                elif self._impl_name == 'iqln':
                    replay_vs = []
                    for v_func in self._impl._value_funcs:
                        replay_vs.append(v_func(replay_observations))
                    replay_vs = torch.mean(torch.stack(replay_vs, dim=0), dim=0)
                old_replay_diff_qs.append(replay_qs - replay_vs)
            old_replay_diff_qs = torch.cat(old_replay_diff_qs, dim=0)

            replay_diff_qs = torch.cat([new_replay_diff_qs, old_replay_diff_qs]).detach().to(torch.float32).to('cpu').squeeze()
            _, indices = torch.sort(replay_diff_qs, descending=True)
            indices = indices[:max_save_num]
            indices_new = indices[indices < len(new_replay_dataset)]
            indices_old = indices[indices >= len(new_replay_dataset)] - len(new_replay_dataset)
            print(str_)
            replay_observations = torch.cat([new_replay_dataset.tensors[0][indices_new], old_replay_dataset.tensors[0][indices_old]], dim=0)
            replay_actions = torch.cat([new_replay_dataset.tensors[1][indices_new], old_replay_dataset.tensors[1][indices_old]], dim=0)
            replay_rewards = torch.cat([new_replay_dataset.tensors[2][indices_new], old_replay_dataset.tensors[2][indices_old]], dim=0)
            replay_next_observations = torch.cat([new_replay_dataset.tensors[3][indices_new], old_replay_dataset.tensors[3][indices_old]], dim=0)
            replay_terminals = torch.cat([new_replay_dataset.tensors[4][indices_new], old_replay_dataset.tensors[4][indices_old]], dim=0)
            replay_policy_actions = torch.cat([new_replay_dataset.tensors[5][indices_new], old_replay_dataset.tensors[5][indices_old]], dim=0)
            replay_qs = torch.cat([new_replay_dataset.tensors[6][indices_new], old_replay_dataset.tensors[6][indices_old]], dim=0)
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
        elif mix_type == 'random':
            return super(new_replay_dataset, old_replay_dataset, dataset_id, max_save_num, mix_type)
        return replay_dataset
