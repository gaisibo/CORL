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
from d3rlpy.algos.combo import COMBO
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    DYNAMICS_NOT_GIVEN_ERROR,
    ActionSpace,
)
from d3rlpy.base import LearnableBase
from d3rlpy.algos.combo import COMBO
from d3rlpy.iterators import TransitionIterator
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer, dynamics_reward_prediction_error_scorer, dynamics_prediction_variance_scorer
from d3rlpy.iterators.random_iterator import RandomIterator
from d3rlpy.iterators.round_iterator import RoundIterator
from d3rlpy.logger import LOG, D3RLPyLogger
import gym

from online.utils import ReplayBuffer
from online.eval_policy import eval_policy

from myd3rlpy.siamese_similar import similar_mb, similar_mb_euclid, similar_phi, similar_psi
# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from myd3rlpy.algos.torch.co_combo_impl import COImpl
from myd3rlpy.algos.co import CO
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class CO(CO, COMBO):
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

    _impl: COImpl
    _actor_learning_rate: float
    _critic_learning_rate: float
    _phi_learning_rate: float
    _psi_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _phi_optim_factory: OptimizerFactory
    _psi_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    # actor必须被重放，没用选择。
    _replay_phi: bool
    _replay_psi: bool
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
    _reduce_replay: str
    _replay_critic: bool
    _replay_model: bool
    _generate_step: int
    _select_time: int
    _model_noise: float

    _task_id: str
    _single_head: bool

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 1e-4,
        phi_learning_rate: float = 1e-4,
        psi_learning_rate: float = 1e-4,
        model_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory : OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        phi_optim_factory: OptimizerFactory = AdamFactory(),
        psi_optim_factory: OptimizerFactory = AdamFactory(),
        model_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        model_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        replay_type='orl',
        phi_bc_loss=True,
        psi_bc_loss=True,
        train_phi=True,
        id_size: int = 7,
        batch_size: int = 1024,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        gem_alpha: float = 1,
        agem_alpha: float = 1,
        ewc_r_walk_alpha: float = 0.5,
        damping: float = 0.1,
        epsilon: float = 0.1,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        initial_temperature: float = 1.0,
        conservative_weight: float = 1.0,
        n_action_samples: int = 10,
        soft_q_backup: bool = False,
        rollout_interval: int = 1000,
        rollout_horizon: int = 1,
        rollout_batch_size: int = 50000,
        real_ratio: float = 0.5,
        generated_maxlen: int = 50000 * 5 * 5,
        target_smoothing_sigma: float = 0.2,
        target_smoothing_clip: float = 0.5,
        alpha: float = 2.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl = None,
        impl_name = 'co',
        # n_train_dynamics = 1,
        retrain_topk = 4,
        log_prob_topk = 10,
        model_n_ensembles = 5,
        experience_type = 'random_transition',
        sample_type = 'retrain',
        reduce_replay = 'retrain',
        use_phi = False,
        use_model = False,
        replay_critic = False,
        replay_model = False,
        generate_step = 100,
        model_noise = 0,
        retrain_time = 1,
        orl_alpha = 1,
        replay_alpha = 1,
        retrain_model_alpha = 1,
        select_time = 30,

        task_id = 0,
        single_head = True,
        **kwargs: Any
    ):
        super().__init__(
            actor_learning_rate = actor_learning_rate,
            critic_learning_rate = critic_learning_rate,
            temp_learning_rate = temp_learning_rate,
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            temp_optim_factory = temp_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            batch_size = batch_size,
            n_frames = n_frames,
            n_steps = n_steps,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            update_actor_interval = update_actor_interval,
            initial_temperature = initial_temperature,
            conservative_weight = conservative_weight,
            n_action_samples = n_action_samples,
            soft_q_backup = soft_q_backup,
            rollout_interval = rollout_interval,
            rollout_horizon = rollout_horizon,
            rollout_batch_size = rollout_batch_size,
            real_ratio = real_ratio,
            generated_maxlen = generated_maxlen,
            target_smoothing_sigma = target_smoothing_sigma,
            target_smoothing_clip = target_smoothing_clip,
            alpha = alpha,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
            impl = impl,
            kwargs = kwargs,
        )
        self._replay_type = replay_type
        self._id_size = id_size

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_r_walk_alpha = ewc_r_walk_alpha
        self._damping = damping
        self._epsilon = epsilon

        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_bc_loss = phi_bc_loss
        self._psi_bc_loss = psi_bc_loss
        self._train_phi = train_phi

        self._impl_name = impl_name
        # self._n_train_dynamics = n_train_dynamics
        self._retrain_topk = retrain_topk
        self._log_prob_topk = log_prob_topk
        self._experience_type = experience_type
        self._sample_type = sample_type
        self._reduce_replay = reduce_replay

        self._task_id = task_id

        self._begin_grad_step = 0

        self._model_learning_rate = model_learning_rate
        self._model_optim_factory = model_optim_factory
        self._model_encoder_factory = model_encoder_factory
        self._model_n_ensembles = model_n_ensembles
        self._retrain_model_alpha = retrain_model_alpha
        self._use_phi = use_phi
        self._use_model = use_model
        self._replay_critic = replay_critic
        self._replay_model = replay_model
        self._generate_step = generate_step
        self._select_time = select_time
        self._model_noise = model_noise
        self._orl_alpha = orl_alpha
        self._retrain_time = retrain_time
        self._replay_alpha = replay_alpha
        self._single_head = single_head

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int, task_id: int
    ) -> None:
        assert self._impl_name in ['co', 'gemco', 'agemco']
        self._impl = COImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            phi_learning_rate=self._phi_learning_rate,
            psi_learning_rate=self._psi_learning_rate,
            model_learning_rate=self._model_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            phi_optim_factory=self._phi_optim_factory,
            psi_optim_factory=self._psi_optim_factory,
            model_optim_factory=self._model_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            model_encoder_factory=self._model_encoder_factory,
            q_func_factory=self._q_func_factory,
            replay_type=self._replay_type,
            gamma=self._gamma,
            gem_alpha=self._gem_alpha,
            agem_alpha=self._agem_alpha,
            ewc_r_walk_alpha=self._ewc_r_walk_alpha,
            damping=self._damping,
            epsilon=self._epsilon,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            conservative_weight=self._conservative_weight,
            n_action_samples=self._n_action_samples,
            real_ratio=self._real_ratio,
            soft_q_backup=self._soft_q_backup,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            model_n_ensembles=self._model_n_ensembles,
            use_phi=self._use_phi,
            use_model=self._use_model,
            replay_critic=self._replay_critic,
            replay_model=self._replay_model,
            replay_alpha=self._replay_alpha,
            retrain_model_alpha=self._retrain_model_alpha,
            # single_head 在impl里面被强制设为False了。
            single_head=self._single_head,
        )
        self._impl.build(task_id)

    def begin_update(self, batch: TransitionMiniBatch) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._begin_update(batch)
        self._begin_grad_step += 1
        return loss

    # 注意欧氏距离最近邻被塞到actions后面了。
    def _begin_update(self, batch: TransitionMiniBatch) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        if self._replay_critic:
            critic_loss = self._impl.begin_update_critic(batch)
            metrics.update({"begin_critic_loss": critic_loss})

        if self._grad_step % self._update_actor_interval == 0:
            actor_loss = self._impl.begin_update_actor(batch)
            metrics.update({"begin_actor_loss": actor_loss})
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]]=None, batch2: TransitionMiniBatch = None) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._update(batch, replay_batches)
        self._grad_step += 1
        return loss

    # 注意欧氏距离最近邻被塞到actions后面了。
    def _update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]]=None) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        if not self._replay_critic:
            critic_loss, _, _ = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})
        else:
            critic_loss, replay_critic_loss, _ = self._impl.update_critic(batch, replay_batches)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})

        if self._grad_step % self._update_actor_interval == 0:
            actor_loss, replay_actor_loss, _ = self._impl.update_actor(batch, replay_batches)
            metrics.update({"actor_loss": actor_loss})
            metrics.update({"replay_actor_loss": replay_actor_loss})
            if self._temp_learning_rate > 0:
                temp_loss, temp = self._impl.update_temp(batch)
                metrics.update({"temp_loss": temp_loss, 'temp': temp})
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, real_action_size, batch_size = 64,
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._impl._policy is not None

        if not self._is_generating_new_data():
            return None

        init_transitions = self._sample_initial_transitions(transitions)
        print(len(init_transitions))

        rets: List[Transition] = []

        # rollout
        i = 0
        while i + batch_size < len(init_transitions):
            batch = TransitionMiniBatch(init_transitions[i: i + batch_size])
            observations = torch.from_numpy(batch.observations).to(self._impl.device)
            rewards = torch.from_numpy(batch.rewards).to(self._impl.device)
            actions = self._impl._policy(observations)
            prev_transitions: List[Transition] = []
            for _ in range(self._get_rollout_horizon()):
                # predict next state
                indexes = torch.randint(len(self._impl._dynamic._models), size=(observations.shape[0],))
                next_observations, next_rewards = self._impl._dynamic(observations[:, :real_observation_size], actions[:, :real_action_size], indexes)

                # sample policy action
                next_actions = self._impl._policy(next_observations)

                # append new transitions
                new_transitions = []
                for j in range(batch_size):
                    transition = Transition(
                        observation_shape=self._impl.observation_shape,
                        action_size=self._impl.action_size,
                        observation=observations[j].cpu().detach().numpy(),
                        action=actions[j].cpu().detach().numpy(),
                        reward=float(rewards[j].cpu().detach().numpy()),
                        next_observation=next_observations[j].cpu().detach().numpy(),
                        terminal=0.0,
                    )

                    if prev_transitions:
                        prev_transitions[j].next_transition = transition
                        transition.prev_transition = prev_transitions[j]

                    new_transitions.append(transition)

                prev_transitions = new_transitions
                rets += new_transitions
                observations = next_observations
                actions = next_actions
            i += batch_size
        batch = TransitionMiniBatch(init_transitions[i:])
        observations = torch.from_numpy(batch.observations).to(self._impl.device)
        rewards = torch.from_numpy(batch.rewards).to(self._impl.device)
        actions = self._impl._policy(observations)
        prev_transitions: List[Transition] = []
        for _ in range(self._get_rollout_horizon()):
            # predict next state
            indexes = torch.randint(len(self._impl._dynamic._models), size=(observations.shape[0],))
            next_observations, next_rewards = self._impl._dynamic(observations[:, :real_observation_size], actions[:, :real_action_size], indexes)

            # sample policy action
            next_actions = self._impl._policy(next_observations)

            # append new transitions
            new_transitions = []
            for j in range(len(init_transitions) - i):
                transition = Transition(
                    observation_shape=self._impl.observation_shape,
                    action_size=self._impl.action_size,
                    observation=observations[j].cpu().detach().numpy(),
                    action=actions[j].cpu().detach().numpy(),
                    reward=float(rewards[j].cpu().detach().numpy()),
                    next_observation=next_observations[j].cpu().detach().numpy(),
                    terminal=0.0,
                )

                if prev_transitions:
                    prev_transitions[j].next_transition = transition
                    transition.prev_transition = prev_transitions[j]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations
            actions = next_actions
        i += batch_size
