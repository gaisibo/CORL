import os
import copy
import sys
import time
import math
import random
from typing import Any, Dict, Optional, Sequence, List, Union, Callable, Tuple, Generator, Iterator, cast
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
from d3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch, Transition
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.cql import CQL
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

from myd3rlpy.siamese_similar import similar_mb, similar_phi, similar_psi
# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from myd3rlpy.algos.torch.co_impl import COImpl


model_base_type = ['model_base_run', 'model_base_same', 'model_base_different']
class CO(CQL):
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
    _temp_learning_rate: float
    _alpha_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _alpha_optim_factory: OptimizerFactory
    _phi_optim_factory: OptimizerFactory
    _psi_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    # actor必须被重放，没用选择。
    _replay_actor_alpha: float
    _replay_critic_alpha: float
    _replay_critic: bool
    _replay_phi: bool
    _replay_psi: bool
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _initial_temperature: float
    _initial_alpha: float
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    # _dynamics: Optional[ProbabilisticEnsembleDynamics]
    _rollout_interval: int
    _rollout_horizon: int
    _rollout_batch_size: int
    _use_gpu: Optional[Device]
    _change_reward: str
    _reduce_replay: str

    _task_id: str

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 1e-4,
        alpha_learning_rate: float = 0,
        phi_learning_rate: float = 1e-4,
        psi_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory : OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        alpha_optim_factory: OptimizerFactory = AdamFactory(),
        phi_optim_factory: OptimizerFactory = AdamFactory(),
        psi_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        replay_actor_alpha = 1,
        replay_critic_alpha = 1,
        replay_phi_alpha = 1,
        replay_psi_alpha = 1,
        replay_type='orl',
        phi_bc_loss=True,
        psi_bc_loss=True,
        train_phi=True,
        id_size: int = 7,
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        gem_gamma: float = 1,
        agem_alpha: float = 1,
        tau: float = 0.005,
        n_critics: int = 2,
        initial_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 10.0,
        n_action_samples: int = 10,
        soft_q_backup: bool =False,
        # dynamics: Optional[ProbabilisticEnsembleDynamics] = None,
        rollout_interval: int = 1000,
        rollout_horizon: int = 5,
        rollout_batch_size: int = 50000,
        real_ratio: float = 0.5,
        generated_maxlen: int = 50000 * 5 * 5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl = None,
        impl_name = 'co',
        # n_train_dynamics = 1,
        phi_topk = 20,
        retrain_topk = 4,
        log_prob_topk = 10,
        generate_type = 'random',
        experience_type = 'random_transition',
        change_reward = 'change',
        reduce_replay = 'retrain',

        task_id = 0,
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
            initial_temperature = initial_temperature,
            conservative_weight = conservative_weight,
            n_action_samples = n_action_samples,
            soft_q_backup = soft_q_backup,
            rollout_interval = rollout_interval,
            rollout_horizon = rollout_horizon,
            rollout_batch_size = rollout_batch_size,
            real_ratio = real_ratio,
            generated_maxlen = generated_maxlen,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
            impl = impl,
            kwargs = kwargs,
        )
        self._replay_type = replay_type
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._id_size = id_size

        self._alpha_optim_factory = alpha_optim_factory
        self._alpha_learning_rate = alpha_learning_rate
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._gem_gamma = gem_gamma
        self._agem_alpha = agem_alpha

        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_bc_loss = phi_bc_loss
        self._psi_bc_loss = psi_bc_loss
        self._train_phi = train_phi
        self._replay_phi_alpha = replay_phi_alpha
        self._replay_psi_alpha = replay_psi_alpha

        self._impl_name = impl_name
        # self._n_train_dynamics = n_train_dynamics
        self._phi_topk = phi_topk
        self._retrain_topk = retrain_topk
        self._log_prob_topk = log_prob_topk
        self._generate_type = generate_type
        self._experience_type = experience_type
        self._change_reward = change_reward
        self._reduce_replay = reduce_replay

        self._rollout_interval = rollout_interval
        self._rollout_horizon = rollout_horizon
        self._rollout_batch_size = rollout_batch_size

        self._task_id = task_id

        self._begin_grad_step = 0

        self._dynamics = None

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
            alpha_learning_rate=self._alpha_learning_rate,
            phi_learning_rate=self._phi_learning_rate,
            psi_learning_rate=self._psi_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            alpha_optim_factory=self._alpha_optim_factory,
            phi_optim_factory=self._phi_optim_factory,
            psi_optim_factory=self._psi_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            replay_critic_alpha=self._replay_critic_alpha,
            replay_actor_alpha=self._replay_actor_alpha,
            replay_type=self._replay_type,
            gamma=self._gamma,
            gem_gamma=self._gem_gamma,
            agem_alpha=self._agem_alpha,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_alpha=self._initial_alpha,
            initial_temperature=self._initial_temperature,
            alpha_threshold=self._alpha_threshold,
            conservative_weight=self._conservative_weight,
            n_action_samples=self._n_action_samples,
            soft_q_backup=self._soft_q_backup,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
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

        critic_loss = self._impl.begin_update_critic(batch)
        metrics.update({"begin_critic_loss": critic_loss})

        actor_loss = self._impl.begin_update_actor(batch)
        metrics.update({"begin_actor_loss": actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]], batch2: TransitionMiniBatch = None) -> Dict[int, float]:
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
    def _update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]]) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        if self._alpha_learning_rate > 0:
            alpha_loss, alpha = self._impl.update_alpha(batch)
            metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

        critic_loss, replay_critic_loss, _ = self._impl.update_critic(batch, replay_batches)
        metrics.update({"critic_loss": critic_loss})
        metrics.update({"replay_critic_loss": replay_critic_loss})

        actor_loss, replay_actor_loss, _ = self._impl.update_actor(batch, replay_batches)
        metrics.update({"actor_loss": actor_loss})
        metrics.update({"replay_actor_loss": replay_actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def _update_phi(self, batch: TransitionMiniBatch):
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        # if self._train_phi:
        phi_loss, phi_diff_phi, phi_diff_r, phi_diff_kl, phi_diff_psi = self._impl.update_phi(batch)
        metrics.update({"phi_loss": phi_loss})
        psi_loss, psi_diff_loss, psi_u_loss = self._impl.update_psi(batch, pretrain=False)
        metrics.update({"psi_loss": psi_loss})
        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def _update_model(self, batch: TransitionMiniBatch):
        assert self._dynamics._impl is not None
        loss = self._dynamics._impl.update(batch)
        return {"loss": loss}

    def fit(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Union[Dict[int, TensorDataset], Dict[int, List[Transition]]]] = None,
        env: gym.envs = None,
        original = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_begin_epochs: Optional[int] = None,
        n_dynamic_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 1,
        eval_freq: int = int(5e3),
	scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        test: bool = False,
        # train_dynamics = False,
    ) -> List[Tuple[int, Dict[int, float]]]:
        """Trains with the given dataset.
        .. code-block:: python
            algo.fit(episodes, n_steps=1000000)
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
            list of result tuples (epoch, metrics) per epoch.
        """
        results = list(
            self.fitter(
                task_id,
                dataset,
                replay_datasets,
                env,
                original,
                seed,
                n_epochs,
                n_begin_epochs,
                n_dynamic_epochs,
                n_steps,
                n_steps_per_epoch,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodes,
                save_interval,
                discount,
                start_timesteps,
                expl_noise,
                eval_freq,
                scorers,
                shuffle,
                callback,
                real_action_size,
                real_observation_size,
                test,
                # train_dynamics,
            )
        )
        return results

    def fitter(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Union[Dict[int, TensorDataset], Dict[int, List[Transition]]]] = None,
        env: gym.envs = None,
        original = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_begin_epochs: Optional[int] = None,
        n_dynamic_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
	scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        test: bool = False,
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
            observation_shape = [real_observation_size]
            self._create_impl(
                self._process_observation_shape(observation_shape), action_size, task_id
            )
            LOG.debug("Models have been built.")
        else:
            self._impl.change_task(task_id)
            LOG.warning("Skip building models since they're already built.")

        # if self._generate_type in model_base_type:
        #     if self._dynamics is not None:
        #         if self._dynamics._impl is None:
        #             action_size = real_action_size
        #             observation_shape = [real_observation_size]
        #             self._dynamics.create_impl(
        #                 observation_shape, action_size
        #             )
        #             LOG.debug("Dynamics have been built.")
        #         else:
        #             LOG.warning("Skip building dynamics since they're already built.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        # add reference to active logger to algo class during fit
        self._active_logger = logger
        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)
        if replay_datasets is not None:
            self._replay_loss_histories = dict()
            for replay_num in self._loss_history:
                self._replay_loss_histories[replay_num] = defaultdict(list)

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
                    if n_epochs is None and n_steps is not None:
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

            # if self._generate_type in model_base_type:
            #     assert self._dynamics is not None
            #     self._dynamics.fit(
            #         transitions,
            #         n_epochs=1,
            #         scorers={
            #            'observation_error': dynamics_observation_prediction_error_scorer,
            #            'reward_error': dynamics_reward_prediction_error_scorer,
            #            'variance': dynamics_prediction_variance_scorer,
            #         },
            #         pretrain=True
            #     )

            if n_epochs is None and n_steps is not None:
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

            if task_id != 0:
                if n_begin_epochs is not None:
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

                total_step = 0
                if n_begin_epochs is not None:
                    for epoch in range(1, n_begin_epochs + 1):

                        # dict to add incremental mean losses to epoch
                        epoch_loss = defaultdict(list)

                        range_gen = tqdm(
                            range(len(iterator)),
                            disable=not show_progress,
                            desc=f"Epoch {epoch}/{n_epochs}",
                        )

                        iterator.reset()

                        for itr in range_gen:

                            with logger.measure_time("step"):
                                # pick transitions
                                with logger.measure_time("sample_batch"):
                                    batch = next(iterator)

                                # update parameters
                                with logger.measure_time("algorithm_update"):
                                    loss = self.begin_update(batch)
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

            total_step = 0
            for epoch in range(1, n_epochs + 1):

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

                for itr in range_gen:

                    # generate new transitions with dynamics models
                    if self._generate_type in model_base_type:
                        new_transitions = self.generate_replay_data(
                            dataset,
                            original,
                            in_task=True,
                            real_action_size=real_action_size,
                            real_observation_size=real_observation_size,
                        )
                        if isinstance(new_transitions, Tuple):
                            new_transitions = new_transitions[0]
                    elif self._generate_type == 'siamese':
                        new_transitions = self.generate_replay_data_phi(
                            dataset,
                            original,
                            in_task=True,
                            real_action_size=real_action_size,
                            real_observation_size=real_observation_size,
                        )
                        if isinstance(new_transitions, Tuple):
                            new_transitions = new_transitions[0]
                    else:
                        new_transitions = None
                        # new_transitions = self.generate_new_data(
                        #     iterator.transitions,
                        #     real_observation_size=real_observation_size,
                        # )

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

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(batch, replay_batches)
                            if self._generate_type == 'siamese':
                                loss_phi = self._update_phi(batch)
                            # self._impl.increase_siamese_alpha(epoch - n_epochs, itr / len(iterator))

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)
                            epoch_loss[name].append(val)
                        if self._generate_type == 'siamese':
                            for name, val in loss_phi.items():
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

            if self._generate_type not in model_base_type and self._experience_type in model_base_type:
                # training loop
                total_step = 0
                for epoch in range(1, n_dynamic_epochs + 1):

                    # dict to add incremental mean losses to epoch
                    epoch_loss = defaultdict(list)

                    range_gen = tqdm(
                        range(len(iterator)),
                        disable=not show_progress,
                        desc=f"Epoch {int(epoch)}/{n_epochs}",
                    )

                    iterator.reset()

                    for batch_num, itr in enumerate(range_gen):
                        if batch_num > 1000 and test:
                            break

                        # generate new transitions with dynamics models

                        with logger.measure_time("step"):
                            # pick transitions
                            with logger.measure_time("sample_batch"):
                                batch = next(iterator)

                            # update parameters
                            with logger.measure_time("algorithm_update"):
                                loss = self._update_model(batch)

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

            if self._generate_type != 'siamese' and self._experience_type == 'siamese':
                total_step = 0
                for epoch in range(1, n_epochs + 1):

                    # dict to add incremental mean losses to epoch
                    epoch_loss = defaultdict(list)

                    range_gen = tqdm(
                        range(len(iterator)),
                        disable=not show_progress,
                        desc=f"Epoch {epoch}/{n_epochs}",
                    )

                    iterator.reset()

                    for itr in range_gen:

                        with logger.measure_time("step"):
                            # pick transitions
                            with logger.measure_time("sample_batch"):
                                batch = next(iterator)

                            # update parameters
                            with logger.measure_time("algorithm_update"):
                                loss= self._update_phi(batch)
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

            # drop reference to active logger since out of fit there is no active
            # logger
            self._active_logger = None

            # for EWC
            if self._replay_type == 'agem':
                self._impl.agem_post_train_process(iterator)
            elif self._replay_type == 'gem':
                self._impl.gem_post_train_process()

        else:
            replay_buffer = ReplayBuffer(real_observation_size, real_action_size)
            # Evaluate untrained policy
            evaluations = [eval_policy(self._impl._policy, env, seed)]

            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num = 0
            if n_steps is None and n_epochs is not None:
                n_steps = n_epochs * n_steps_per_epoch
            else:
                assert n_steps is not None

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])

            kwargs = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "max_action": max_action,
                    "discount": discount,
                    "tau": self._tau,
            }

            for t in range(n_steps):

                episode_timesteps += 1

                # Select action randomly or according to policy
                if t < start_timesteps:
                    action = env.action_space.sample()
                else:
                    action = (
                            self._impl._policy(np.array(state))
                            + np.random.normal(0, max_action * expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)

                # Perform action
                next_state, reward, done, _ = env.step(action)
                done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

                # Store data in replay buffer
                replay_buffer.add(state, action, next_state, reward, done_bool)

                state = next_state
                episode_reward += reward

                batch = replay_buffer.sample()
                # Train agent after collecting sufficient data
                if t >= start_timesteps:
                    # update parameters
                    with logger.measure_time("sample_batch"):
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
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch, replay_batches)

                if done:
                    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                    print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                    # Reset environment
                    state, done = env.reset(), False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                # Evaluate episode
                if (t + 1) % eval_freq == 0:
                    evaluations.append(eval_policy(self._impl._policy, env, seed))
                    # save metrics
                    metrics = logger.commit(t, n_steps)

                    # save model parameters
                    if t % save_interval == 0:
                        logger.save_model(t, self)
                    if scorers and eval_episodes:
                        self._evaluate(eval_episodes, scorers, logger)


    def test(
        self,
        save_dir: str,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Union[Dict[int, TensorDataset], Dict[int, List[Transition]]]] = None,
        env: gym.envs = None,
        original = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_begin_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 1,
        eval_freq: int = int(5e3),
	scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        # train_dynamics = False,
    ) -> List[Tuple[int, Dict[int, float]]]:
        """Trains with the given dataset.
        .. code-block:: python
            algo.fit(episodes, n_steps=1000000)
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
            list of result tuples (epoch, metrics) per epoch.
        """
        results = list(
            self.tester(
                save_dir,
                task_id,
                dataset,
                replay_datasets,
                env,
                original,
                seed,
                n_epochs,
                n_begin_epochs,
                n_steps,
                n_steps_per_epoch,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodes,
                save_interval,
                discount,
                start_timesteps,
                expl_noise,
                eval_freq,
                scorers,
                shuffle,
                callback,
                real_action_size,
                real_observation_size,
                # train_dynamics,
            )
        )
        return results

    def tester(
        self,
        save_dir: str,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Union[Dict[int, TensorDataset], Dict[int, List[Transition]]]] = None,
        env: gym.envs = None,
        original = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_begin_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
	scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        # train_dynamics: bool = False,
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
            observation_shape = [real_observation_size]
            self._create_impl(
                self._process_observation_shape(observation_shape), action_size, task_id
            )
            LOG.debug("Models have been built.")
        else:
            self._impl.change_task(task_id)
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

        # add reference to active logger to algo class during fit
        self._active_logger = logger
        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)
        if replay_datasets is not None:
            self._replay_loss_histories = dict()
            for replay_num in self._loss_history:
                self._replay_loss_histories[replay_num] = defaultdict(list)

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
                    if n_epochs is None and n_steps is not None:
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

            # if self._generate_type == 'model_base':
            #     assert self._dynamics is not None
            #     assert origin_transitions is not None
            #     if train_dynamics:
            #         self._dynamics.fit(
            #             origin_transitions,
            #             n_epochs=1,
            #             scorers={
            #                'observation_error': dynamics_observation_prediction_error_scorer,
            #                'reward_error': dynamics_reward_prediction_error_scorer,
            #                'variance': dynamics_prediction_variance_scorer,
            #             },
            #             pretrain=True
            #         )

            if n_epochs is None and n_steps is not None:
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

            if task_id != 0:
                if n_begin_epochs is not None:
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

                total_step = 0
                # if self._generate_type == 'model_base':
                #     assert self._dynamics is not None
                #     self._dynamics._network = self
                if n_begin_epochs is not None:
                    for epoch in range(1, n_begin_epochs + 1):

                        # dict to add incremental mean losses to epoch
                        epoch_loss = defaultdict(list)

                        total_step += len(iterator)

                        iterator.reset()

                        if epoch % save_interval == 0:
                            self._impl.load_model(os.path.join(save_dir, total_step))

                        if scorers and eval_episodes:
                            self._evaluate(eval_episodes, scorers, logger)

                        # save metrics
                        metrics = logger.commit(epoch, str(total_step))

                        yield epoch, metrics

            total_step = 0
            for epoch in range(1, n_epochs + 1):

                # dict to add incremental mean losses to epoch
                epoch_loss = defaultdict(list)

                range_gen = tqdm(
                    range(len(iterator)),
                    disable=not show_progress,
                    desc=f"Epoch {epoch}/{n_epochs}",
                )

                iterator.reset()

                total_step += len(iterator)

                if epoch % save_interval == 0:
                    self._impl.load_model(os.path.join(save_dir, str(total_step)))

                if scorers and eval_episodes:
                    self._evaluate(eval_episodes, scorers, logger)

                # save metrics
                metrics = logger.commit(epoch, total_step)

                yield epoch, metrics

            # drop reference to active logger since out of fit there is no active
            # logger
            self._active_logger = None

        else:

            for t in range(n_steps):

                # Evaluate episode
                if (t + 1) % eval_freq == 0:

                    # save model parameters
                    if t % save_interval == 0:
                        self._impl.load_model(os.path.join(save_dir, str(t)))
                    if scorers and eval_episodes:
                        self._evaluate(eval_episodes, scorers, logger)

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._dynamics, DYNAMICS_NOT_GIVEN_ERROR

        if not self._is_generating_new_data():
            return None

        init_transitions = self._sample_initial_transitions(transitions)

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = self._sample_rollout_action(observations)
        rewards = batch.rewards
        prev_transitions: List[Transition] = []

        for _ in range(self._get_rollout_horizon()):

            # predict next state
            pred = self._dynamics.predict(observations[:, :real_observation_size], actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, next_rewards, variances = pred

            # regularize by uncertainty
            next_observations, next_rewards = self._mutate_transition(
                next_observations, next_rewards, variances
            )

            # sample policy action
            next_actions = self._sample_rollout_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(len(init_transitions)):
                transition = Transition(
                    observation_shape=self._impl.observation_shape,
                    action_size=self._impl.action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i],
                    terminal=0.0,
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()
            rewards = next_rewards.copy()

        return rets

    def generate_new_data_trajectory(self, dataset, original_index, in_task=False, max_export_time = 100, max_reward=None, real_action_size=1, real_observation_size=1):
        assert self._impl is not None
        if not self._is_generating_new_data():
            return None
        # 关键算法
        start_indexes = np.array([original_index])
        prev_transition = None
        replay_indexes = None
        new_transitions = []

        transitions = [transition for episode in dataset.episodes for transition in episode]
        transition_observations = np.stack([transition.observation for transition in transitions])
        transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)

        export_time = 0
        while start_indexes.shape[0] and export_time < max_export_time:
            start_observations = torch.from_numpy(dataset._observations[start_indexes]).to(self._impl.device)
            start_actions = self._impl._policy(start_observations)
            start_rewards = dataset._rewards[start_indexes]

            mus, logstds = [], []
            for model in self._dynamics._impl._dynamics._models:
                mu, logstd = model.compute_stats(start_observations[:, :real_observation_size], start_actions)
                mus.append(mu)
                logstds.append(logstd)
            mus = torch.stack(mus, dim=1)
            logstds = torch.stack(logstds, dim=1)
            mus = mus[torch.arange(start_observations.shape[0]), torch.randint(len(self._dynamics._impl._dynamics._models), size=(start_observations.shape[0],))]
            logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(self._dynamics._impl._dynamics._models), size=(start_observations.shape[0],))]
            dist = Normal(mus, torch.exp(logstds))
            pred = dist.rsample()
            pred_observations = pred[:, :-1]
            next_x = start_observations + pred_observations
            next_action = self._impl._policy(next_x)
            next_reward = pred[:, -1].view(-1, 1)

            for i in range(start_observations.shape[0]):
                transition = Transition(
                    observation_shape = self._impl.observation_shape,
                    action_size = self._impl.action_size,
                    observation = start_observations[i].cpu().numpy(),
                    action = start_actions[i].cpu().detach().numpy(),
                    reward = start_rewards[i],
                    next_observation = next_x[i].cpu().detach().numpy(),
                    terminal = 0,
                )
                new_transitions.append(transition)

            if start_indexes.shape[0] > 0:
                line_indexes = dataset._actions[start_indexes[0], real_action_size:].astype(np.int64)
                near_indexes, _, _ = similar_mb(mus[0], logstds[0], transition_observations[line_indexes, :real_observation_size], np.expand_dims(dataset._rewards, axis=1), self._dynamics._impl._dynamics, topk=self._phi_topk, input_indexes=line_indexes)
            else:
                near_indexes, _, _ = similar_mb(mus[0], logstds[0], transition_observations[:, :real_observation_size], np.expand_dims(dataset._rewards, axis=1), self._dynamics._impl._dynamics, topk=self._phi_topk)
            start_indexes = near_indexes
            if replay_indexes is not None:
                start_indexes = np.setdiff1d(start_indexes, replay_indexes, True)
            start_rewards = dataset._rewards[start_indexes]
            if max_reward is not None:
                start_indexes = start_indexes[start_rewards >= max_reward]
            if start_indexes.shape[0] == 0:
                break
            if replay_indexes is not None:
                replay_indexes = np.concatenate([replay_indexes, start_indexes], axis=0)
            else:
                replay_indexes = start_indexes
            export_time += 1

        random.shuffle(new_transitions)
        if self._replay_type != 'bc' or in_task:
            return new_transitions, None
        elif self._replay_type == 'bc' and not in_task:
            replay_observations = torch.cat([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0).detach()
            replay_actions = torch.cat([torch.from_numpy(transition.action) for transition in new_transitions], dim=0).detach()
            replay_rewards = torch.cat([torch.from_numpy(transition.reward) for transition in new_transitions], dim=0).detach()
            replay_next_observations = torch.cat([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0).detach()
            replay_next_actions = torch.cat([torch.from_numpy(transition.next_action) for transition in new_transitions], dim=0).detach()
            replay_next_rewards = torch.cat([torch.from_numpy(transition.next_reward) for transition in new_transitions], dim=0).detach()
            replay_terminals = torch.cat([torch.from_numpy(transition.terminals) for transition in new_transitions], dim=0).detach()
            replay_dists = self._impl._policy.dist(replay_observations)
            replay_means, replay_std_logs = replay_dists.mean.detach(), replay_dists.stddev.detach()
            replay_qs = self._impl.q_func(replay_observations, replay_actions).detach()
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals, replay_means, replay_std_logs, replay_qs)
            return replay_dataset

    def generate_replay_data_reduce(self, orl_indexes, orl_transitions, real_action_size):
        assert self._impl is not None
        assert self._impl._q_func is not None
        assert self._impl._policy is not None
        choosed_samples = []
        for indexes, transitions in zip(orl_indexes, orl_transitions):
            diffs = []
            transition_batch = TransitionMiniBatch(transitions.tolist())
            transition_q = self._impl._q_func(torch.from_numpy(transition_batch.observations).to(self._impl.device), torch.from_numpy(transition_batch.actions[:, :real_action_size]).to(self._impl.device))
            transition_dist = self._impl._policy.dist(torch.from_numpy(transition_batch.observations).to(self._impl.device))
            for i in range(len(transitions)):
                reduce_transitions = []
                for j in range(len(transitions)):
                    if j != i:
                        reduce_transitions.append(transitions[j])
                    else:
                        reduce_transition = Transition(
                            observation_shape=transitions[i].get_observation_shape(),
                            action_size=transitions[i].get_action_size(),
                            observation=transitions[i].observation,
                            action=transitions[i].action,
                            reward=-10000,
                            next_observation=transitions[i].next_observation,
                            terminal=transitions[i].terminal,
                            next_transition = transitions[i].next_transition,
                            prev_transition = transitions[i].prev_transition,
                        )
                        reduce_transitions.append(transitions[i])
                        reduce_transitions.append(reduce_transition)
                reduce_batch = TransitionMiniBatch(reduce_transitions)
                impl_copy = self._impl.copy_weight()
                self._update(reduce_batch, None)
                reduce_q = self._impl._q_func(torch.from_numpy(transition_batch.observations).to(self._impl.device), torch.from_numpy(transition_batch.actions[:, :real_action_size]).to(self._impl.device))
                reduce_dist = self._impl._policy.dist(torch.from_numpy(transition_batch.observations).to(self._impl.device))
                self._impl.reload_weight(impl_copy)
                diff = torch.mean(torch.abs(transition_q - reduce_q) + torch.distributions.kl.kl_divergence(transition_dist, reduce_dist))
                diffs.append(diff)
            diffs = torch.stack(diffs, dim=0)
            if diffs.shape[0] > self._retrain_topk:
                _, choosed_sample_index = torch.topk(diffs, k=self._retrain_topk)
            else:
                _, choosed_sample_index = torch.topk(diffs, k=diffs.shape[0])
            choosed_sample_index = choosed_sample_index.cpu().detach().numpy()
            choosed_samples.append(indexes[choosed_sample_index])
        return choosed_samples

    def generate_replay_data_phi(self, dataset, original_indexes, max_save_num=1000, max_export_time=1000, max_reward=None, real_action_size=1, real_observation_size=1, low_log_prob=0.8, in_task=False):
        assert self._impl is not None
        assert self._impl._policy is not None
        assert self._impl._q_func is not None
        if in_task:
            if not self._is_generating_new_data():
                return None

        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset
        # 关键算法

        orl_indexes = []
        for original_index in original_indexes:
            start_indexes = np.array([original_index])

            transitions = np.array([transition for episode in dataset.episodes for transition in episode])
            transition_observations = np.stack([transition.observation for transition in transitions])
            transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
            transition_actions = np.stack([transition.action for transition in transitions])
            transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
            transition_rewards = np.stack([transition.reward for transition in transitions])
            transition_rewards = torch.from_numpy(transition_rewards).to(self._impl.device)

            export_time = 0
            replay_indexes = []
            while start_indexes.shape[0] != 0 and export_time < max_export_time and len(replay_indexes) < max_save_num:
                start_observations = torch.from_numpy(dataset._observations[start_indexes]).to(self._impl.device)
                start_actions = self._impl._policy(start_observations)
                start_rewards = dataset._rewards[start_indexes]

                indexes_euclid = np.array(dataset._actions[start_indexes, real_action_size:], dtype=np.int64)
                near_observations = dataset._observations[indexes_euclid]
                near_actions = dataset._actions[indexes_euclid][:, :, :real_action_size]

                near_indexes_list = []
                if start_indexes.shape[0] > 0:
                    near_indexes, _, _ = similar_phi(start_observations, start_actions[:, :real_action_size], near_observations, near_actions, self._impl._phi, indexes_euclid, topk=self._phi_topk)
                    for i in range(near_indexes.shape[0]):
                        near_indexes_list.append(near_indexes[i])
                near_indexes_list.reverse()
                # 附近的所有点都会留下来作为orl的数据集。
                for start_indexes in near_indexes_list:
                    orl_indexes.append(start_indexes.astype(np.int64))
                # 第一个非空的将作为接下来衍生的起点被保留，同时也用作replay的数据集。
                for start_indexes in near_indexes_list:
                    new_start_indexes = np.setdiff1d(start_indexes, replay_indexes, True)
                    if new_start_indexes.shape[0] != 0:
                        start_indexes = new_start_indexes
                    else:
                        continue
                if start_indexes is None:
                    break

                start_rewards = transition_rewards[start_indexes]
                if max_reward is not None:
                    start_indexes = start_indexes[start_rewards >= max_reward]
                replay_indexes = np.concatenate([replay_indexes, start_indexes], axis=0)
                export_time += 1

            orl_transitions = [transitions[orl_index] for orl_index in orl_indexes]

            # 用log_prob缩小范围。
            orl_indexes_ = []
            for log_prob_indexes, log_prob_transitions in zip(orl_indexes, orl_transitions):
                transition_batch = TransitionMiniBatch(log_prob_transitions.tolist())
                transition_dists = self._impl._policy.dist(torch.from_numpy(transition_batch.observations).to(self._impl.device))
                transition_log_prob = torch.mean(transition_dists.log_prob(torch.from_numpy(transition_batch.actions[:, :real_action_size]).to(self._impl.device)), dim=1)
                if self._reduce_replay == 'retrain' and not in_task:
                    if transition_log_prob.shape[0] > self._log_prob_topk:
                        _, choosed_sample_index = torch.topk(transition_log_prob, k=self._log_prob_topk)
                        choosed_sample_index = log_prob_indexes[choosed_sample_index.cpu().detach().numpy()]
                    else:
                        _, choosed_sample_index = torch.topk(transition_log_prob, k=transition_log_prob.shape[0])
                        choosed_sample_index = log_prob_indexes[choosed_sample_index.cpu().detach().numpy()]
                else:
                    if transition_log_prob.shape[0] > self._retrain_topk:
                        _, choosed_sample_index = torch.topk(transition_log_prob, k=self._retrain_topk)
                        choosed_sample_index = log_prob_indexes[choosed_sample_index.cpu().detach().numpy()]
                    else:
                        _, choosed_sample_index = torch.topk(transition_log_prob, k=transition_log_prob.shape[0])
                        choosed_sample_index = log_prob_indexes[choosed_sample_index.cpu().detach().numpy()]
                orl_indexes_.append(choosed_sample_index)
            orl_indexes = orl_indexes_
            orl_transitions = [transitions[orl_index] for orl_index in orl_indexes]

            if self._reduce_replay == 'retrain' and not in_task:
                orl_indexes = self.generate_replay_data_reduce(orl_indexes, orl_transitions, real_action_size)

        orl_indexes = np.concatenate(orl_indexes, axis=0)
        orl_indexes = np.unique(orl_indexes)
        orl_transitions = [transitions[orl_index] for orl_index in orl_indexes]
        if not in_task:
            assert self._impl is not None
            assert self._impl._policy is not None
            if self._change_reward == 'change':
                assert self._impl._q_func is not None
                orl_transitions_ = []
                orl_observations = torch.from_numpy(np.stack([orl_transition.observation for orl_transition in orl_transitions], axis=0)).to(self._impl.device)
                orl_actions = torch.from_numpy(np.stack([orl_transition.action for orl_transition in orl_transitions], axis=0)).to(self._impl.device)
                orl_next_observations = torch.from_numpy(np.stack([orl_transition.next_observation for orl_transition in orl_transitions], axis=0)).to(self._impl.device)
                orl_next_actions = self._impl._policy(orl_next_observations)
                orl_terminals = torch.from_numpy(np.stack([orl_transition.terminal for orl_transition in orl_transitions], axis=0)).to(self._impl.device)
                orl_q = self._impl._q_func(orl_observations, orl_actions[:, :real_action_size]).squeeze()
                orl_diff_q = orl_q - self.gamma * self._impl._q_func(orl_next_observations, orl_next_actions[:, :real_action_size]).squeeze()
                print(f'orl_terminals: {orl_terminals.shape}')
                print(f'orl_diff_q: {orl_diff_q.shape}')
                print(f'orl_q: {orl_q.shape}')
                orl_rewards = torch.where(orl_terminals == 0, orl_diff_q, orl_q)
                for i, transition in enumerate(orl_transitions):
                    orl_transition = Transition(
                        observation_shape=transition.get_observation_shape(),
                        action_size=transition.get_action_size(),
                        observation=transition.observation,
                        action=transition.action,
                        reward=orl_rewards[i].cpu().detach().numpy(),
                        next_observation=transition.next_observation,
                        next_transition = transitions[i].next_transition,
                        prev_transition = transitions[i].prev_transition,
                        terminal=transition.terminal,
                    )
                    orl_transitions_.append(orl_transition)
                orl_transitions = orl_transitions_
        random.shuffle(orl_transitions)
        if len(orl_transitions) > max_save_num:
            orl_transitions = orl_transitions[:max_save_num]
        if in_task:
            return orl_transitions, None

        replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in orl_transitions], dim=0)
        replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in orl_transitions], dim=0)
        replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in orl_transitions], dim=0)
        replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in orl_transitions], dim=0)
        replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in orl_transitions], dim=0)
        if self._replay_type != 'bc' or in_task:
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
            return orl_transitions, replay_dataset
        elif self._replay_type == 'bc' and not in_task:
            replay_dists = self._impl._policy.dist(replay_observations.to(self._impl.device))
            replay_means, replay_std_logs = replay_dists.mean.detach(), replay_dists.stddev.detach()
            replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_means, replay_std_logs, replay_qs, replay_phis, replay_psis)
            return replay_dataset, replay_dataset

    def generate_replay_data_transition(self, dataset, in_task=False, max_save_num=1000, real_action_size=1):
        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset
        transitions = [transition for episode in episodes for transition in episode.transitions]
        if self._experience_type == 'random_transition':
            random.shuffle(transitions)
        elif self._experience_type == 'max_reward':
            transitions = sorted(transitions, key=lambda x: x.reward, reversed=True)
        elif self._experience_type == 'max_match':
            for transition in transitions:
                transition_dists = self._impl._policy.dist(torch.from_numpy(transition.observations).to(self._impl.device))
                transition.log_prob = transition_dists.log_prob(torch.from_numpy(transition.actions[:, :real_action_size]).to(self._impl.device)).sum(dim=1)
            transitions = sorted(transitions, key=lambda x: x.log_prob, reversed=True)
        else:
            raise NotImplementedError
        transitions = transitions[:max_save_num]
        if in_task:
            if not self._is_generating_new_data():
                return None
        new_transitions = []
        for transition in transitions:
            new_transitions.append(
                transition
            )

        replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0)
        replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in new_transitions], dim=0)
        replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in new_transitions], dim=0)
        replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0)
        replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in new_transitions], dim=0)
        if self._replay_type != 'bc' or in_task:
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
            return new_transitions, replay_dataset
        elif self._replay_type == 'bc' and not in_task:
            replay_dists = self._impl._policy.dist(replay_observations.to(self._impl.device))
            replay_means, replay_std_logs = replay_dists.mean.detach(), replay_dists.stddev.detach()
            replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_means, replay_std_logs, replay_qs, replay_phis, replay_psis)
            return replay_dataset, replay_dataset

    def generate_replay_data_episode(self, dataset, in_task=False, max_save_num=1000, real_action_size=1):
        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset
        if self._experience_type == 'random_episode':
            random.shuffle(episodes)
        elif self._experience_type == 'max_reward_end':
            episodes = sorted(episodes, key=lambda x: x.rewards[-1], reversed=True)
        elif self._experience_type == 'max_reward_mean':
            episodes = sorted(episodes, key=lambda x: sum(x.rewards) / len(x.rewards), reversed=True)
        elif self._experience_type in ['max_match_end', 'max_match_mean']:
            for episode in episodes:
                for transition in episode.transitions:
                    transition_dists = self._impl._policy.dist(torch.from_numpy(transition.observations).to(self._impl.device))
                    transition.log_prob = transition_dists.log_prob(torch.from_numpy(transition.actions[:, :real_action_size]).to(self._impl.device)).sum(dim=1)
            if self._experience_type == 'max_match_end':
                episodes = sorted(trainsitions, key=lambda x: x.log_prob[-1], reversed=True)
            elif self._experience_type == 'max_match_mean':
                episodes = sorted(trainsitions, key=lambda x: sum(x.log_prob) / len(x.log_prob), reversed=True)
        else:
            raise NotImplementedError
        transitions = [transition for episode in episodes for transition in episode.transitions]
        transitions = transitions[:max_save_num]
        if in_task:
            if not self._is_generating_new_data():
                return None
        new_transitions = []
        for transition in transitions:
            new_transitions.append(
                transition
            )

        replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0)
        replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in new_transitions], dim=0)
        replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in new_transitions], dim=0)
        replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0)
        replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in new_transitions], dim=0)
        if self._replay_type != 'bc' or in_task:
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
            return new_transitions, replay_dataset
        elif self._replay_type == 'bc' and not in_task:
            replay_dists = self._impl._policy.dist(replay_observations.to(self._impl.device))
            replay_means, replay_std_logs = replay_dists.mean.detach(), replay_dists.stddev.detach()
            replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_means, replay_std_logs, replay_qs, replay_phis, replay_psis)
            return replay_dataset, replay_dataset

    def generate_replay_data_episode(self, dataset, in_task=False, max_save_num=1000, real_action_size=1):
        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset
        if self._experience_type == 'random_episode':
            random.shuffle(episodes)
        elif self._experience_type == 'max_reward_end':
            episodes = sorted(episodes, key=lambda x: x.rewards[-1], reversed=True)
        elif self._experience_type == 'max_reward_mean':
            episodes = sorted(episodes, key=lambda x: sum(x.rewards) / len(x.rewards), reversed=True)
        elif self._experience_type in ['max_match_end', 'max_match_mean']:
            for episode in episodes:
                for transition in episode.transitions:
                    transition_dists = self._impl._policy.dist(torch.from_numpy(transition.observations).to(self._impl.device))
                    transition.log_prob = transition_dists.log_prob(torch.from_numpy(transition.actions[:, :real_action_size]).to(self._impl.device)).sum(dim=1)
            if self._experience_type == 'max_match_end':
                episodes = sorted(trainsitions, key=lambda x: x.log_prob[-1], reversed=True)
            elif self._experience_type == 'max_match_mean':
                episodes = sorted(trainsitions, key=lambda x: sum(x.log_prob) / len(x.log_prob), reversed=True)
        else:
            raise NotImplementedError
        transitions = [transition for episode in episodes for transition in episode.transitions]
        transitions = transitions[:max_save_num]
        if in_task:
            if not self._is_generating_new_data():
                return None
        new_transitions = []
        for transition in transitions:
            new_transitions.append(
                transition
            )

        replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0)
        replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in new_transitions], dim=0)
        replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in new_transitions], dim=0)
        replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0)
        replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in new_transitions], dim=0)
        if self._replay_type != 'bc' or in_task:
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
            return new_transitions, replay_dataset
        elif self._replay_type == 'bc' and not in_task:
            replay_dists = self._impl._policy.dist(replay_observations.to(self._impl.device))
            replay_means, replay_std_logs = replay_dists.mean.detach(), replay_dists.stddev.detach()
            replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
            replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_means, replay_std_logs, replay_qs, replay_phis, replay_psis)
            return replay_dataset, replay_dataset

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0
