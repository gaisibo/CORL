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
from d3rlpy.iterators import TransitionIterator
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer, dynamics_reward_prediction_error_scorer, dynamics_prediction_variance_scorer
from d3rlpy.iterators.random_iterator import RandomIterator
from d3rlpy.iterators.round_iterator import RoundIterator
from d3rlpy.logger import LOG, D3RLPyLogger
import gym

from online.utils import ReplayBuffer
from online.eval_policy import eval_policy

from myd3rlpy.siamese_similar import similar_mb
# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from myd3rlpy.algos.torch.co_combo_impl import COCOMBOImpl as COImpl
from myd3rlpy.algos.co import CO
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
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
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
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
        model_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory : OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        model_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        model_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        replay_type='orl',
        id_size: int = 7,
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        gem_alpha: float = 1,
        agem_alpha: float = 1,
        ewc_rwalk_alpha: float = 0.5,
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
        clone_actor = True,
        replay_critic = False,
        replay_model = False,
        generate_step = 100,
        model_noise = 0,
        retrain_time = 1,
        orl_alpha = 1,
        replay_alpha = 1,
        retrain_model_alpha = 1,
        select_time = 100,

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
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon

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
        self._clone_actor = clone_actor
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
            model_learning_rate=self._model_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            model_optim_factory=self._model_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            model_encoder_factory=self._model_encoder_factory,
            q_func_factory=self._q_func_factory,
            replay_type=self._replay_type,
            gamma=self._gamma,
            gem_alpha=self._gem_alpha,
            agem_alpha=self._agem_alpha,
            ewc_rwalk_alpha=self._ewc_rwalk_alpha,
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
            clone_actor=self._clone_actor,
            replay_critic=self._replay_critic,
            replay_model=self._replay_model,
            replay_alpha=self._replay_alpha,
            retrain_model_alpha=self._retrain_model_alpha,
            # single_head 在impl里面被强制设为False了。
            single_head=self._single_head,
        )
        self._impl.build(task_id)

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

        if not self._replay_model:
            model_loss, _, _ = self._impl.update_model(batch)
            metrics.update({"model_loss": model_loss})
        else:
            model_loss, replay_model_loss, _ = self._impl.update_model(batch, replay_batches)
            metrics.update({"model_loss": model_loss})
            metrics.update({"replay_model_loss": replay_model_loss})

        if not self._replay_critic:
            critic_loss, _, _ = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})
        else:
            critic_loss, replay_critic_loss, _ = self._impl.update_critic(batch, replay_batches)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})

        if self._grad_step % self._update_actor_interval == 0:
            actor_loss, replay_actor_loss, replay_clone_loss, _ = self._impl.update_actor(batch, replay_batches)
            metrics.update({"actor_loss": actor_loss})
            metrics.update({"replay_actor_loss": replay_actor_loss})
            metrics.update({"replay_clone_loss": replay_clone_loss})
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

    def _update_model(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]] = None):
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        if not self._replay_model:
            model_loss = self._impl.only_update_model(batch)
            metrics.update({"model_loss": model_loss})
        else:
            model_loss, replay_model_loss, _ = self._impl.update_model(batch, replay_batches)
            metrics.update({"model_loss": model_loss})
            metrics.update({"replay_model_loss": replay_model_loss})
        return metrics

    def fitter(
        self,
        task_id: str,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Union[Dict[int, TensorDataset], Dict[int, List[Transition]]]] = None,
        env: gym.envs = None,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = 500000,
        n_steps_per_epoch: int = 5000,
        # pretrain_state_dict: Optional[Dict[str, Any]] = None,
        # pretrain_task_id: Optional[int] = None,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = False,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
	scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        test: bool = False,
        epoch_num: Optional[int] = None,
    ) -> Generator[Tuple[int, Dict[int, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.
        .. code-block:: python
            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)
        Args:
            dataset: list of episodes to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            iterator yielding current epoch and metrics dict.
        """
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = real_action_size
            print(f'real_action_size: {real_action_size}')
            observation_shape = [real_observation_size]
            self._create_impl(
                self._process_observation_shape(observation_shape), action_size, task_id
            )
            self._impl._impl_id = task_id
            self._impl.clone_networks()
            LOG.debug("Models have been built.")
        else:
            self._impl.change_task(task_id)
            env.set_task_idx(int(task_id))
            self._impl.rebuild_critic()
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        self._active_logger = logger

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        if replay_datasets is not None:
            replay_dataloaders: Optional[Dict[int, DataLoader]]
            replay_iterators: Optional[Dict[int, Iterator]]
            replay_dataloaders = dict()
            replay_iterators = dict()
            for replay_num, replay_dataset in replay_datasets.items():
                if isinstance(replay_dataset, TensorDataset):
                    dataloader = DataLoader(replay_dataset, batch_size=self._batch_size, shuffle=True)
                    replay_dataloaders[replay_num] = dataloader
                    replay_iterators[replay_num] = iter(replay_dataloaders[replay_num])
                else:
                    if n_steps is not None:
                        assert n_steps >= n_steps_per_epoch
                        n_epochs = n_steps // n_steps_per_epoch
                        iterator = RandomIterator(
                            replay_dataset,
                            n_steps_per_epoch,
                            batch_size=self._batch_size,
                            n_steps=self._n_steps,
                            gamma=self._gamma,
                            n_frames=self._n_frames,
                            real_ratio=self._real_ratio,
                            generated_maxlen=self._generated_maxlen,
                        )
                        LOG.debug("RandomIterator is selected.")
                    elif n_epochs is not None and n_steps is None:
                        iterator = RoundIterator(
                            replay_dataset,
                            batch_size=self._batch_size,
                            n_steps=self._n_steps,
                            gamma=self._gamma,
                            n_frames=self._n_frames,
                            real_ratio=self._real_ratio,
                            generated_maxlen=self._generated_maxlen,
                            shuffle=shuffle,
                        )
                        LOG.debug("RoundIterator is selected.")
                    else:
                        raise ValueError("Either of n_epochs or n_steps must be given.")
                    replay_iterators[replay_num] = iterator
        else:
            replay_dataloaders = None
            replay_iterators = None

        iterator: TransitionIterator
        if env is None:
            assert dataset is not None
            transitions = []
            if isinstance(dataset, MDPDataset):
                for episode in cast(MDPDataset, dataset).episodes:
                    transitions += episode.transitions
            elif not dataset:
                raise ValueError("empty dataset is not supported.")
            elif isinstance(dataset[0], Episode):
                for episode in cast(List[Episode], dataset):
                    transitions += episode.transitions
            elif isinstance(dataset[0], Transition):
                transitions = list(cast(List[Transition], dataset))
            else:
                raise ValueError(f"invalid dataset type: {type(dataset)}")

            # initialize scaler
            if self._scaler:
                LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
                self._scaler.fit(transitions)

            # initialize action scaler
            if self._action_scaler:
                LOG.debug(
                    "Fitting action scaler...",
                    action_scaler=self._action_scaler.get_type(),
                )
                self._action_scaler.fit(transitions)

            # initialize reward scaler
            if self._reward_scaler:
                LOG.debug(
                    "Fitting reward scaler...",
                    reward_scaler=self._reward_scaler.get_type(),
                )
                self._reward_scaler.fit(transitions)
        if n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = RandomIterator(
                transitions,
                n_steps_per_epoch,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                transitions,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")
        total_step = 0
        print(f'train policy')
        for epoch in range(1, n_epochs + 1):
            if epoch > 1 and test:
                break

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {epoch}/{n_epochs}",
            )

            iterator.reset()
            if replay_dataloaders is not None:
                replay_iterators = dict()
                for replay_num, replay_dataloader in replay_dataloaders.items():
                    replay_iterators[replay_num] = iter(replay_dataloader)
            else:
                replay_iterators = None

            for batch_num, itr in enumerate(range_gen):
                if batch_num > 10 and test:
                    break
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)
                        if not self._clone_finish:
                            if replay_iterators is not None:
                                assert replay_dataloaders is not None
                                replay_batches = dict()
                                for replay_iterator_num in replay_iterators.keys():
                                    try:
                                        replay_batches[replay_iterator_num] = next(replay_iterators[replay_iterator_num])
                                    except StopIteration:
                                        replay_iterators[replay_iterator_num] = iter(replay_dataloaders[replay_iterator_num])
                                        replay_batches[replay_iterator_num] = next(replay_iterators[replay_iterator_num])
                            else:
                                replay_batches = None
                        else:
                            replay_batches = None

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch, replay_batches)
                        # self._impl.increase_siamese_alpha(epoch - n_epochs, itr / len(iterator))

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save loss to loss history dict
            if epoch_num is None:
                self._loss_history["epoch"].append(epoch)
            else:
                self._loss_history["epoch"].append(epoch_num)
            self._loss_history["step"].append(total_step)
            for name, vals in epoch_loss.items():
                if vals:
                    self._loss_history[name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # for EWC
        if self._replay_type in ['rwalk', 'ewc']:
            self._impl.ewc_rwalk_post_train_process(iterator)
        elif self._replay_type == 'si':
            self._impl.si_post_train_process()
        elif self._replay_type == 'gem':
            self._impl.gem_post_train_process()
        elif self._clone_actor and self._replay_type == 'bc':
            self._impl.bc_post_train_process()
        elif self._replay_type == 'fix':
            self._impl.fix_post_train_process()

        if self._clone_finish:
            total_step = 0
            print(f'clone policy')
            for epoch in range(1, n_epochs + 1):
                if epoch > 1 and test:
                    break

                # dict to add incremental mean losses to epoch
                epoch_loss = defaultdict(list)

                range_gen = tqdm(
                    range(len(iterator)),
                    disable=not show_progress,
                    desc=f"Epoch {epoch}/{n_epochs}",
                )

                iterator.reset()
                if replay_dataloaders is not None:
                    replay_iterators = dict()
                    for replay_num, replay_dataloader in replay_dataloaders.items():
                        replay_iterators[replay_num] = iter(replay_dataloader)
                else:
                    replay_iterators = None

                for batch_num, itr in enumerate(range_gen):
                    if batch_num > 10 and test:
                        break

                    new_transitions = self.generate_new_data(transitions=iterator.transitions, real_observation_size=real_observation_size, real_action_size=real_action_size)
                    if new_transitions:
                        iterator.add_generated_transitions(new_transitions)
                        LOG.debug(
                            f"{len(new_transitions)} transitions are generated.",
                            real_transitions=len(iterator.transitions),
                            fake_transitions=len(iterator.generated_transitions),
                        )
                    with logger.measure_time("step"):
                        # pick transitions
                        with logger.measure_time("sample_batch"):
                            batch = next(iterator)
                            if not self._clone_finish:
                                if replay_iterators is not None:
                                    assert replay_dataloaders is not None
                                    replay_batches = dict()
                                    for replay_iterator_num in replay_iterators.keys():
                                        try:
                                            replay_batches[replay_iterator_num] = next(replay_iterators[replay_iterator_num])
                                        except StopIteration:
                                            replay_iterators[replay_iterator_num] = iter(replay_dataloaders[replay_iterator_num])
                                            replay_batches[replay_iterator_num] = next(replay_iterators[replay_iterator_num])
                                else:
                                    replay_batches = None
                            else:
                                replay_batches = None

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self._replay_update(batch, replay_batches)
                            # self._impl.increase_siamese_alpha(epoch - n_epochs, itr / len(iterator))

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)
                            epoch_loss[name].append(val)

                        # update progress postfix with losses
                        if itr % 10 == 0:
                            mean_loss = {
                                k: np.mean(v) for k, v in epoch_loss.items()
                            }
                            range_gen.set_postfix(mean_loss)

                    total_step += 1

                    # call callback if given
                    if callback:
                        callback(self, epoch, total_step)

                # save loss to loss history dict
                self._loss_history["epoch"].append(epoch)
                self._loss_history["step"].append(total_step)
                for name, vals in epoch_loss.items():
                    if vals:
                        self._loss_history[name].append(np.mean(vals))

                if scorers and eval_episodes:
                    self._evaluate(eval_episodes, scorers, logger)

                # save metrics
                metrics = logger.commit(epoch, total_step)

                # save model parameters
                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                yield epoch, metrics