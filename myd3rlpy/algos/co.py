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

from myd3rlpy.models.encoders import LargeVectorEncoderFactory
from myd3rlpy.siamese_similar import similar_mb, similar_phi, similar_psi
# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from myd3rlpy.algos.torch.co_impl import COImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class CO(TD3PlusBC):
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
    _model_noise: float

    _task_id: str

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-3,
        critic_learning_rate: float = 3e-3,
        phi_learning_rate: float = 1e-4,
        psi_learning_rate: float = 1e-4,
        model_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory : OptimizerFactory = AdamFactory(),
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
        generate_step = 0,
        model_noise = 0,
        retrain_time = 10,
        orl_alpha = 1,
        replay_alpha = 1,

        task_id = 0,
        **kwargs: Any
    ):
        super().__init__(
            actor_learning_rate = actor_learning_rate,
            critic_learning_rate = critic_learning_rate,
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            batch_size = batch_size,
            n_frames = n_frames,
            n_steps = n_steps,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
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

        self._dynamics = None
        self._model_learning_rate = model_learning_rate
        self._model_optim_factory = model_optim_factory
        self._model_encoder_factory = model_encoder_factory
        self._model_n_ensembles = model_n_ensembles
        self._use_phi = use_phi
        self._use_model = use_model
        self._replay_critic = replay_critic
        self._replay_model = replay_model
        self._generate_step = generate_step
        self._model_noise = model_noise
        self._orl_alpha = orl_alpha
        self._retrain_time = retrain_time
        self._replay_alpha = replay_alpha

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int, task_id: int
    ) -> None:
        assert self._impl_name in ['co', 'gemco', 'agemco']
        self._impl = COImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            phi_learning_rate=self._phi_learning_rate,
            psi_learning_rate=self._psi_learning_rate,
            model_learning_rate=self._model_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
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
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            alpha=self._alpha,
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


        if not self._replay_critic:
            critic_loss = self._impl.only_update_critic(batch)
            metrics.update({"critic_loss": critic_loss})
        else:
            critic_loss, replay_critic_loss, _ = self._impl.update_critic(batch, replay_batches)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})

        if self._grad_step % self._update_actor_interval == 0:
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

    def _update_model(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]] = None):
        assert self._impl is not None
        metrics = {}

        if not self._replay_model:
            model_loss = self._impl.only_update_model(batch)
            metrics.update({"model_loss": model_loss})
        else:
            model_loss, replay_model_loss, _ = self._impl.update_model(batch, replay_batches)
            metrics.update({"model_loss": model_loss})
            metrics.update({"replay_model_loss": replay_model_loss})
        return metrics

    def fit(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Union[Dict[int, TensorDataset], Dict[int, List[Transition]]]] = None,
        env: gym.envs = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_begin_epochs: Optional[int] = None,
        n_dynamic_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        n_dynamic_steps: Optional[int] = None,
        n_dynamic_steps_per_epoch: int = 10000,
        n_begin_steps: Optional[int] = None,
        n_begin_steps_per_epoch: int = 10000,
        dynamic_state_dict: Optional[Dict[str, Any]] = None,
        pretrain_state_dict: Optional[Dict[str, Any]] = None,
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
                seed,
                n_epochs,
                n_begin_epochs,
                n_dynamic_epochs,
                n_steps,
                n_steps_per_epoch,
                n_begin_steps,
                n_begin_steps_per_epoch,
                n_dynamic_steps,
                n_dynamic_steps_per_epoch,
                dynamic_state_dict,
                pretrain_state_dict,
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
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_begin_epochs: Optional[int] = None,
        n_dynamic_epochs: Optional[int] = None,
        n_steps: Optional[int] = 500000,
        n_steps_per_epoch: int = 5000,
        n_begin_steps: Optional[int] = 500000,
        n_begin_steps_per_epoch: int = 5000,
        n_dynamic_steps: Optional[int] = 500000,
        n_dynamic_steps_per_epoch: int = 5000,
        dynamic_state_dict: Optional[Dict[str, Any]] = None,
        pretrain_state_dict: Optional[Dict[str, Any]] = None,
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
            print(f'real_action_size: {real_action_size}')
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

            if n_dynamic_epochs is None and n_dynamic_steps is not None:
                assert n_dynamic_steps >= n_dynamic_steps_per_epoch
                n_dynamic_epochs = n_dynamic_steps // n_dynamic_steps_per_epoch
                dynamic_iterator = RandomIterator(
                    transitions,
                    n_dynamic_steps_per_epoch,
                    batch_size=self._batch_size,
                    n_steps=self._n_steps,
                    gamma=self._gamma,
                    n_frames=self._n_frames,
                    real_ratio=self._real_ratio,
                    generated_maxlen=self._generated_maxlen,
                )
            elif n_dynamic_epochs is not None and n_dynamic_steps is None:
                dynamic_iterator = RoundIterator(
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

            # update model
            if dynamic_state_dict is not None:
                for key, value in dynamic_state_dict.items():
                    print(f'dynamic: {key}: {value.shape}')
                print()
                for key, value in self._impl.named_parameters():
                    print(f'impl: {key}: {value.shape}')
                assert False
                for key, value in dynamic_state_dict.items():
                    try:
                        obj = getattr(self._impl, key)
                        if isinstance(obj, (torch.nn.Module)):
                            obj = getattr(self._impl, key)
                            obj.load_state_dict(dynamic_state_dict[key])
                    except:
                        key = str(key)
                        obj = getattr(self._impl, key)
                        if isinstance(obj, (torch.nn.Module)):
                            obj = getattr(self._impl, key)
                            obj.load_state_dict(dynamic_state_dict[key])
            else:
                total_step = 0
                for epoch in range(1, n_dynamic_epochs + 1):
                    if epoch > 3 and test:
                        break

                    # dict to add incremental mean losses to epoch
                    epoch_loss = defaultdict(list)

                    range_gen = tqdm(
                        range(len(dynamic_iterator)),
                        disable=not show_progress,
                        desc=f"Epoch {epoch}/{n_dynamic_epochs}",
                    )

                    dynamic_iterator.reset()

                    for batch_num, itr in enumerate(range_gen):
                        if batch_num > 1000 and test:
                            break

                        with logger.measure_time("step"):
                            # pick transitions
                            with logger.measure_time("sample_batch"):
                                batch = next(dynamic_iterator)

                            # update parameters
                            with logger.measure_time("algorithm_update"):
                                loss = self._update_model(batch)
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

                    # save loss to loss history dict
                    self._loss_history["epoch"].append(epoch)
                    self._loss_history["step"].append(total_step)
                    for name, vals in epoch_loss.items():
                        if vals:
                            self._loss_history[name].append(np.mean(vals))

                    # save metrics
                    metrics = logger.commit(epoch, total_step)

                    # save model parameters
                    if epoch % save_interval == 0:
                        logger.save_model(total_step, self)

                    yield epoch, metrics

            if task_id == 0 and pretrain_state_dict is not None:
                for key, value in pretrain_state_dict.items():
                    try:
                        obj = getattr(self._impl, key)
                        if isinstance(obj, (torch.nn.Module)):
                            obj = getattr(self._impl, key)
                            obj.load_state_dict(pretrain_state_dict[key])
                    except:
                        print(f'error key: {key}')
                        obj = getattr(self._impl, key)
                        print(obj.state_dict()['state'].keys())
                        print()
                        print(pretrain_state_dict[key]['state'].keys())
                        obj.load_state_dict(pretrain_state_dict[key])
                return

            else:
                if n_begin_epochs is None and n_begin_steps is not None:
                    assert n_begin_steps >= n_begin_steps_per_epoch
                    n_begin_epochs = n_begin_steps // n_begin_steps_per_epoch
                    begin_iterator = RandomIterator(
                        transitions,
                        n_begin_steps_per_epoch,
                        batch_size=self._batch_size,
                        n_steps=self._n_steps,
                        gamma=self._gamma,
                        n_frames=self._n_frames,
                        real_ratio=self._real_ratio,
                        generated_maxlen=self._generated_maxlen,
                    )
                elif n_begin_epochs is not None and n_begin_steps is None:
                    begin_iterator = RoundIterator(
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
                if n_begin_epochs is not None:
                    for epoch in range(1, n_begin_epochs + 1):
                        if epoch > 3 and test:
                            break

                        # dict to add incremental mean losses to epoch
                        epoch_loss = defaultdict(list)

                        range_gen = tqdm(
                            range(len(begin_iterator)),
                            disable=not show_progress,
                            desc=f"Epoch {epoch}/{n_epochs}",
                        )

                        begin_iterator.reset()

                        for batch_num, itr in enumerate(range_gen):
                            if batch_num > 1000 and test:
                                break

                            with logger.measure_time("step"):
                                # pick transitions
                                with logger.measure_time("sample_batch"):
                                    batch = next(begin_iterator)

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
            total_step = 0
            for epoch in range(1, n_epochs + 1):
                if epoch > 3 and test:
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
                    if batch_num > 1000 and test:
                        break

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
                                    if self._generate_step > 0:
                                        replay_batch = dict(zip(replay_name[:-2], replay_batches[replay_iterator_num]))
                                        replay_batch = Struct(**replay_batches[replay_iterator_num])
                                        replay_batch = cast(TorchMiniBatch, replay_batch)
                                        replay_batch = self.generate_new_data(replay_batch, self._impl._observation_shape, self._impl._action_size)
                                        replay_batches[replay_iterator_num] = replay_batch
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

            if self._experience_type == 'siamese':
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
            if self._replay_type in ['r_walk', 'ewc']:
                self._impl.ewc_r_walk_post_train_process(iterator)
            elif self._replay_type == 'si':
                self._impl.si_post_train_process
            elif self._replay_type == 'gem':
                self._impl.gem_post_train_process

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
        n_dynamic_steps: Optional[int] = None,
        n_dynamic_steps_per_epoch: int = 10000,
        n_begin_steps: Optional[int] = None,
        n_begin_steps_per_epoch: int = 10000,
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
        self, batch: TransitionMiniBatch, real_observation_size, real_action_size
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._dynamics, DYNAMICS_NOT_GIVEN_ERROR

        rets: List[Transition] = []

        # rollout
        observations = batch.observations
        actions = self._impl._policy(observations)
        rewards = batch.rewards
        prev_transitions: List[Transition] = []

        with torch.no_grad():
            for _ in range(self._generate_step):

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
                for i in range(batch.observations.shape[0]):
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
            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in rets], dim=0)
            replay_observations = torch.cat([batch.observations, replay_observations], dim=0)
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in rets], dim=0)
            replay_actions = torch.cat([batch.actions, replay_actions], dim=0)
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in rets], dim=0)
            replay_rewards = torch.cat([batch.rewards, replay_rewards], dim=0)
            replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in rets], dim=0)
            replay_next_observations = torch.cat([batch.next_observations, replay_next_observations], dim=0)
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in rets], dim=0)
            replay_terminals = torch.cat([batch.terminals, replay_terminals], dim=0)
            if self._replay_type != 'bc':
                rets = [replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals]
                return rets
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device))
                replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
                    rets = [replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis]
                else:
                    rets = [replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs]
                return rets

    def generate_replay_data_trajectory(self, dataset, original_indexes, max_save_num=1000, max_export_time=100, max_export_step=1000, max_reward=None, real_action_size=1, real_observation_size=1, low_log_prob=0.8, n_epochs=None, n_steps=500000,n_steps_per_epoch=5000, shuffle=True, save_metrics=True, experiment_name=None, with_timestamp=True, logdir='d3rlpy_logs', verbose=True, tensorboard_dir=None):
        assert self._impl is not None
        assert self._impl._policy is not None
        assert self._impl._q_func is not None

        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset
        # 关键算法

        transitions = np.array([transition for episode in dataset.episodes for transition in episode])
        transition_observations = np.stack([transition.observation for transition in transitions])
        transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
        transition_actions = np.stack([transition.action for transition in transitions])
        transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
        transition_rewards = np.stack([transition.reward for transition in transitions])
        transition_rewards = torch.from_numpy(transition_rewards).to(self._impl.device)

        if self._sample_type == 'retrain':
            logger = self._prepare_logger(
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                tensorboard_dir,
            )
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
        iterator.reset()

        orl_indexes = original_indexes
        orl_steps = [0 for _ in original_indexes]
        orl_ns = [0 for _ in original_indexes]
        for orl_index, orl_step, orl_n in zip(orl_indexes, orl_steps, orl_ns):
            start_index = orl_index

            export_time = 0
            epoch_loss = defaultdict(list)
            while len(orl_indexes) < max_save_num:
                export_step = orl_step + 1
                if self._sample_type == 'retrain':
                    result_observations = []
                    result_actions = []
                    result_rewards = []
                    result_next_observations = []
                    result_terminals = []
                while export_step < max_export_step:
                    start_observations = torch.from_numpy(dataset._observations[start_index]).to(self._impl.device).unsqueeze(dim=0)
                    start_actions = self._impl._policy(start_observations)
                    if self._sample_type == 'noise':
                        noise = 0.1 * max(1, (export_time / max_export_time)) * torch.randn(start_actions.shape, device=self._impl.device)
                        start_actions += noise

                    indexes_euclid = np.array(dataset._actions[start_index, real_action_size:], dtype=np.int64)
                    near_observations = dataset._observations[indexes_euclid]
                    near_actions = dataset._actions[indexes_euclid][:, :real_action_size]
                    near_rewards = dataset._rewards[indexes_euclid]

                    if self._experience_type == 'model':
                        mus, logstds = [], []
                        for model in self._impl._dynamic._models:
                            mu, logstd = model.compute_stats(start_observations, start_actions)
                            mus.append(mu)
                            logstds.append(logstd)
                        mus = torch.stack(mus, dim=1)
                        mus += self._model_noise * torch.randn(mus.shape, device=self._impl.device)
                        logstds = torch.stack(logstds, dim=1)
                        if self._sample_type == 'noise':
                            noise = 0.1 * max(1, (export_time / max_export_time))
                            logstds += noise
                        elif self._sample_type == 'retrain':
                            start_next_observations = model(start_observations, start_actions)[0]

                            with torch.no_grad():
                                # original_reward
                                start_rewards = self._impl._q_func(start_observations, start_actions) - self.gamma * self._impl._q_func._compute_target(start_observations).detach()
                                start_rewards -= self._orl_alpha * orl_n
                                # exploration reward

                            start_terminals = torch.zeros(start_observations.shape[0])
                            result_observations.append(start_observations)
                            result_actions.append(start_actions)
                            result_rewards.append(start_rewards)
                            result_next_observations.append(start_next_observations)
                            result_terminals.append(start_terminals)
                        mus = mus[torch.arange(start_observations.shape[0]), torch.randint(len(self._impl._dynamic._models), size=(start_observations.shape[0],))]
                        logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(self._impl._dynamic._models), size=(start_observations.shape[0],))]

                    if self._experience_type == 'siamese':
                        near_index, _, _ = similar_phi(start_observations, start_actions[:, :real_action_size], near_observations, near_actions, self._impl._phi, input_indexes=indexes_euclid, topk=1)
                    elif self._experience_type == 'model':
                        near_index = similar_mb(mus, logstds, near_observations, np.expand_dims(near_rewards, axis=1), topk=1, input_indexes=indexes_euclid)
                    else:
                        raise NotImplementedError
                    start_index = near_index.astype(np.int64)
                    if near_index.astype(np.int64) in orl_indexes:
                        new_index = orl_indexes.index(start_index)
                        orl_ns[new_index] += 1
                        orl_steps[new_index] = min(orl_steps[new_index], export_step)
                    else:
                        orl_indexes.append(start_index)
                        orl_ns.append(1)
                        orl_steps.append(export_step)
                    orl_indexes.append(start_index.astype(np.int64))
                    orl_indexes = list(set(orl_indexes))
                    export_step += 1
                export_time += 1
                if self._sample_type == 'retrain':
                    for _ in range(self._retrain_time):
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            iterator.reset()
                            batch = next(iterator)
                        batch.observations = torch.cat([batch.observations] + result_observations, dim=0)
                        batch.actions = torch.cat([batch.actions] + result_actions, dim=0)
                        batch.rewards = torch.cat([batch.rewards] + result_rewards, dim=0)
                        batch.next_observations = torch.cat([batch.next_observations] + result_next_observations, dim=0)
                        batch.terminals = torch.cat([batch.terminals] + result_terminals, dim=0)
                        loss = self.update(batch)
                        for name, val in loss.items():
                            logger.add_metric(name, val)
                            epoch_loss[name].append(val)
                    # save loss to loss history dict
                    self._loss_history["epoch"].append(epoch)
                    self._loss_history["step"].append(total_step)
                    for name, vals in epoch_loss.items():
                        if vals:
                            self._loss_history[name].append(np.mean(vals))

                    # save metrics
                    metrics = logger.commit(epoch, total_step)

        random.shuffle(orl_indexes)
        orl_indexes = orl_indexes[:max_save_num]
        orl_transitions = [transitions[orl_index] for orl_index in orl_indexes]

        with torch.no_grad():
            assert self._impl is not None
            assert self._impl._policy is not None

            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in orl_transitions], dim=0)
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in orl_transitions], dim=0)
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in orl_transitions], dim=0)
            replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in orl_transitions], dim=0)
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in orl_transitions], dim=0)
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return orl_transitions, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device))
                replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset

    def generate_replay_data_transition(self, dataset, max_save_num=1000, real_action_size=1, batch_size=16):
        with torch.no_grad():
            if isinstance(dataset, MDPDataset):
                episodes = dataset.episodes
            else:
                episodes = dataset
            transitions = [transition for episode in episodes for transition in episode.transitions]
            if self._experience_type == 'random_transition':
                random.shuffle(transitions)
            elif self._experience_type == 'max_reward':
                transitions = sorted(transitions, key=lambda x: x.reward, reverse=True)
            elif self._experience_type == 'min_reward':
                transitions = sorted(transitions, key=lambda x: x.reward)
            elif 'match' in self._experience_type:
                transition_observations = np.stack([transition.observation for transition in transitions])
                transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                transition_actions = np.stack([transition.action for transition in transitions])
                transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                transition_dists = self._impl._policy(transition_observations)
                transition_log_probs = torch.sum((transition_dists - transition_actions) ** 2)
                if self._experience_type == 'max_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1], reverse=True)]
                if self._experience_type == 'min_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1])]
            elif 'model' in self._experience_type:
                transition_rewards = np.stack([transition.reward for transition in transitions])
                transition_rewards = torch.from_numpy(transition_rewards).to(self._impl.device)
                transition_next_observations = np.stack([transition.next_observation for transition in transitions])
                transition_next_observations = torch.from_numpy(transition_next_observations).to(self._impl.device)
                mus, logstds = [], []
                i = 0
                mu = []
                logstd = []
                while i + batch_size < transition_next_observations.shape[0]:
                    mu_batches = []
                    logstd_batches = []
                    for model in self._impl._dynamic._models:
                        transition_observations = np.stack([transition.observation for transition in transitions[i : i + batch_size]])
                        transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                        transition_actions = np.stack([transition.action for transition in transitions[i : i + batch_size]])
                        transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                        mu_batch, logstd_batch = model.compute_stats(transition_observations, transition_actions)
                        mu_batches.append(mu_batch)
                        logstd_batches.append(logstd_batch)
                    mu_batch = sum(mu_batches) / len(mu_batches)
                    logstd_batch = sum(logstd_batches) / len(logstd_batches)
                    mu.append(mu_batch)
                    logstd.append(logstd_batch)
                if i < transition_next_observations.shape[0]:
                    mu_batches = []
                    logstd_batches = []
                    for model in self._impl._dynamic._models:
                        transition_observations = np.stack([transition.observation for transition in transitions[i:]])
                        transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                        transition_actions = np.stack([transition.action for transition in transitions[i:]])
                        transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                        mu_batch, logstd_batch = model.compute_stats(transition_observations, transition_actions)
                        mu_batches.append(mu_batch)
                        logstd_batches.append(logstd_batch)
                    mu_batch = sum(mu_batches) / len(mu_batches)
                    logstd_batch = sum(logstd_batches) / len(logstd_batches)
                    mu.append(mu_batch)
                    logstd.append(logstd_batch)
                mu = torch.cat(mu, dim=0)
                logstd = torch.cat(logstd, dim=0)
                dists = Normal(mu, torch.exp(logstd))
                transition_log_probs = dists.log_prob(torch.cat([transition_next_observations, transition_rewards.unsqueeze(dim=1)], dim=1))
                if self._experience_type == 'max_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1], reverse=True)]
                if self._experience_type == 'min_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1])]
            else:
                raise NotImplementedError
            transitions = transitions[:max_save_num]
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
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return new_transitions, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device))
                replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0

    def _get_rollout_horizon(self):
        return self._rollout_horizon

    def generate_replay_data_episode(self, dataset, max_save_num=1000, real_action_size=1, batch_size=16):
        with torch.no_grad():
            if isinstance(dataset, MDPDataset):
                episodes = dataset.episodes
            else:
                episodes = dataset
            if self._experience_type == 'random_episode':
                random.shuffle(episodes)
            elif self._experience_type == 'max_reward_end':
                episodes = sorted(episodes, key=lambda x: x.rewards[-1], reverse=True)
            elif self._experience_type == 'min_reward_end':
                episodes = sorted(episodes, key=lambda x: x.rewards[-1])
            elif self._experience_type == 'max_reward_mean':
                episodes = sorted(episodes, key=lambda x: sum(x.rewards) / len(x.rewards), reverse=True)
            elif self._experience_type == 'min_reward_mean':
                episodes = sorted(episodes, key=lambda x: sum(x.rewards) / len(x.rewards))
            elif self._experience_type[4:] in ['match_end', 'match_mean']:
                episode_log_probs = []
                for episode in episodes:
                    transitions = episode.transitions
                    transition_observations = np.stack([transition.observation for transition in transitions])
                    transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                    transition_actions = np.stack([transition.action for transition in transitions])
                    transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                    transition_dists = self._impl._policy(transition_observations)
                    transition_log_probs = torch.sum((transition_dists - transition_actions) ** 2)
                    if self._experience_type[4:] == 'match_end':
                        episode_log_probs.append(transition_log_probs[-1])
                    elif self._experience_type[4:] == 'max_match_mean':
                        episode_log_probs.append(torch.mean(transition_log_probs))
                if self._experience_type[:3] == 'max':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_log_probs), key=lambda x: x[1], reverse=True)]
                elif self._experience_type[:3] == 'min':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_log_probs), key=lambda x: x[1])]
            elif self._experience_type[4:] in ['model_end', 'model_mean']:
                episode_log_probs = []
                for episode in episodes:
                    transitions = episode.transitions
                    transition_rewards = np.stack([transition.reward for transition in transitions])
                    transition_rewards = torch.from_numpy(transition_rewards).to(self._impl.device)
                    transition_next_observations = np.stack([transition.next_observation for transition in transitions])
                    transition_next_observations = torch.from_numpy(transition_next_observations).to(self._impl.device)
                    mus, logstds = [], []
                    i = 0
                    mu = []
                    logstd = []
                    while i + batch_size < transition_next_observations.shape[0]:
                        mu_batches = []
                        logstd_batches = []
                        for model in self._impl._dynamic._models:
                            transition_observations = np.stack([transition.observation for transition in transitions[i : i + batch_size]])
                            transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                            transition_actions = np.stack([transition.action for transition in transitions[i : i + batch_size]])
                            transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                            mu_batch, logstd_batch = model.compute_stats(transition_observations, transition_actions)
                            mu_batches.append(mu_batch)
                            logstd_batches.append(logstd_batch)
                        mu_batch = sum(mu_batches) / len(mu_batches)
                        logstd_batch = sum(logstd_batches) / len(logstd_batches)
                        mu.append(mu_batch)
                        logstd.append(logstd_batch)
                    if i < transition_next_observations.shape[0]:
                        mu_batches = []
                        logstd_batches = []
                        for model in self._impl._dynamic._models:
                            transition_observations = np.stack([transition.observation for transition in transitions[i:]])
                            transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                            transition_actions = np.stack([transition.action for transition in transitions[i:]])
                            transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                            mu_batch, logstd_batch = model.compute_stats(transition_observations, transition_actions)
                            mu_batches.append(mu_batch)
                            logstd_batches.append(logstd_batch)
                        mu_batch = sum(mu_batches) / len(mu_batches)
                        logstd_batch = sum(logstd_batches) / len(logstd_batches)
                        mu.append(mu_batch)
                        logstd.append(logstd_batch)
                    mu = torch.cat(mu, dim=0)
                    logstd = torch.cat(logstd, dim=0)
                    dists = Normal(mu, torch.exp(logstd))
                    transition_log_probs = dists.log_prob(torch.cat([transition_next_observations, transition_rewards.unsqueeze(dim=1)], dim=1))
                    if self._experience_type[4:] == 'model_end':
                        episode_log_probs.append(transition_log_probs[-1])
                    elif self._experience_type[4:] == 'model_mean':
                        episode_log_probs.append(torch.mean(transition_log_probs))
                if self._experience_type[:3] == 'max':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_log_probs), key=lambda x: x[1], reverse=True)]
                elif self._experience_type[:3] == 'min':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_log_probs), key=lambda x: x[1])]
            else:
                raise NotImplementedError
            transitions = [transition for episode in episodes for transition in episode.transitions]
            transitions = transitions[:max_save_num]
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
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return new_transitions, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device))
                replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach()
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach()
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset
