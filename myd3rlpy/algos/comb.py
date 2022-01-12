import sys
import time
import math
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
from d3rlpy.algos.combo import COMBO
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    DYNAMICS_NOT_GIVEN_ERROR,
    ActionSpace,
)
from d3rlpy.base import TransitionIterator, TransitionMiniBatch, LearnableBase
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer, dynamics_reward_prediction_error_scorer, dynamics_prediction_variance_scorer
from d3rlpy.iterators.random_iterator import RandomIterator
from d3rlpy.iterators.round_iterator import RoundIterator
from d3rlpy.dynamics import DynamicsBase
from d3rlpy.logger import LOG, D3RLPyLogger
import gym

from online.utils import ReplayBuffer
from online.eval_policy import eval_policy

from myd3rlpy.siamese_similar import similar_mb

class COMB(COMBO):
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
    _temp_learning_rate: float
    _alpha_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _alpha_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    # actor必须被重放，没用选择。
    _replay_actor_alpha: float
    _replay_critic_alpha: float
    _replay_critic: bool
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _initial_temperature: float
    _initial_alpha: float
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    _dynamics: Optional[DynamicsBase]
    _rollout_interval: int
    _rollout_horizon: int
    _rollout_batch_size: int
    _use_gpu: Optional[Device]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 1e-4,
        alpha_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        alpha_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        replay_actor_alpha = 1,
        replay_critic_alpha = 1,
        cql_loss=False,
        q_bc_loss=True,
        td3_loss=False,
        policy_bc_loss=True,
        id_size: int = 7,
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        initial_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 1.0,
        n_action_samples: int = 10,
        soft_q_backup: bool =False,
        dynamics: Optional[DynamicsBase] = None,
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
        origin = None,
        n_train_dynamics = 1,
        topk = 4,
        mb_generate = True,
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
            dynamics = dynamics,
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
        self._cql_loss = cql_loss
        self._q_bc_loss = q_bc_loss
        self._td3_loss = td3_loss
        self._policy_bc_loss = policy_bc_loss
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._id_size = id_size

        self._impl_name = impl_name
        self._origin = origin
        self._n_train_dynamics = n_train_dynamics
        self._topk = topk
        self._use_mb_generate = mb_generate

        self._alpha_optim_factory = alpha_optim_factory
        self._alpha_learning_rate = alpha_learning_rate
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        assert self._impl_name in ['co', 'gemco', 'agemco']
        COImpl = None
        if self._impl_name == 'co':
            from myd3rlpy.algos.torch.comb_impl import COMBImpl as COImpl
        elif self._impl_name == 'gemco':
            from myd3rlpy.algos.torch.gemco_impl import GEMCOImpl as COImpl
        elif self._impl_name == 'agemco':
            from myd3rlpy.algos.torch.agemco_impl import AGEMCOImpl as COImpl
        assert COImpl is not None
        self._impl = COImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            alpha_learning_rate=self._alpha_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            alpha_optim_factory=self._alpha_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            replay_critic_alpha=self._replay_critic_alpha,
            replay_actor_alpha=self._replay_actor_alpha,
            cql_loss=self._cql_loss,
            q_bc_loss=self._q_bc_loss,
            td3_loss=self._td3_loss,
            policy_bc_loss=self._policy_bc_loss,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            initial_alpha=self._initial_alpha,
            alpha_threshold=self._alpha_threshold,
            conservative_weight=self._conservative_weight,
            n_action_samples=self._n_action_samples,
            real_ratio=self._real_ratio,
            soft_q_backup=self._soft_q_backup,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]]) -> Dict[int, float]:
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

    def fit(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        origin_dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Dict[int, List[TensorDataset]]] = None,
        env: gym.envs = None,
        original = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodess: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 10,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 1,
        eval_freq: int = int(5e3),
        scorers: Optional[
            Dict[str, Callable[[int, int], Callable[[Any, List[Episode]], float]]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, Iterator], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        test: bool = False,
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
                origin_dataset,
                replay_datasets,
                env,
                original,
                seed,
                n_epochs,
                n_steps,
                n_steps_per_epoch,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodess,
                save_interval,
                discount,
                start_timesteps,
                expl_noise,
                eval_freq,
                scorers,
                replay_scorers,
                shuffle,
                callback,
                real_action_size,
                real_observation_size,
                test,
            )
        )
        return results

    def fitter(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        origin_dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        replay_datasets: Optional[Optional[Dict[int, List[TensorDataset]]]] = None,
        env: gym.envs = None,
        original = None,
        seed: int = None,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodess: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
        scorers: Optional[
            Dict[str, Callable[[int, int], Callable[[Any, List[Episode]], float]]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, Iterator], float]]
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
            observation_shape = [real_observation_size + self._id_size]
            self.create_impl(
                observation_shape, action_size
            )
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        if self._dynamics is not None:
            if self._dynamics._impl is None:
                action_size = real_action_size
                observation_shape = [real_observation_size]
                self._dynamics.create_impl(
                    observation_shape, action_size
                )
                LOG.debug("Dynamics have been built.")
            else:
                LOG.warning("Skip building dynamics since they're already built.")

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

        replay_dataloaders: Optional[Dict[int, List[TensorDataset]]]
        if replay_datasets is not None:
            replay_dataloaders = dict()
            for replay_num, replay_dataset in replay_datasets.items():
                dataloader = DataLoader(replay_dataset, batch_size=self._batch_size, shuffle=True)
                replay_dataloaders[replay_num] = dataloader
            replay_iterators = dict()
            for replay_num, replay_dataloader in replay_dataloaders.items():
                replay_iterators[replay_num] = iter(replay_dataloader)
        else:
            replay_dataloaders = None
            replay_iterators = None

        iterator: TransitionIterator
        if env is None:

            transitions = []
            if isinstance(dataset, MDPDataset):
                for episode in dataset.episodes:
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

            origin_transitions = []
            if isinstance(origin_dataset, MDPDataset):
                for episode in origin_dataset.episodes:
                    origin_transitions += episode.transitions
            elif not origin_dataset:
                raise ValueError("empty origin_dataset is not supported.")
            elif isinstance(origin_dataset[0], Episode):
                for episode in cast(List[Episode], dataset):
                    origin_transitions += episode.transitions
            elif isinstance(origin_dataset[0], Transition):
                origin_transitions = list(cast(List[Transition], dataset))
            else:
                raise ValueError(f"invalid origin_dataset type: {type(dataset)}")
            assert self._dynamics is not None
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
                LOG.debug("RandomIterator is selected.")
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
                LOG.debug("RoundIterator is selected.")
            else:
                raise ValueError("Either of n_epochs or n_steps must be given.")

            total_step = 0
            # pretrain
            # self._dynamics.fit(
            #     origin_episodes,
            #     n_epochs=100 if not test else 1,
            #     scorers={
            #        'observation_error': dynamics_observation_prediction_error_scorer,
            #        'reward_error': dynamics_reward_prediction_error_scorer,
            #        'variance': dynamics_prediction_variance_scorer,
            #     },
            #     pretrain=True,
            # )
            self._dynamics._network = self
            for epoch in range(1, n_epochs + 1):

                # if self._n_train_dynamics % epoch == 0:
                #     self._dynamics.fit(
                #         origin_episodes,
                #         n_epochs=1,
                #         scorers={
                #            'observation_error': dynamics_observation_prediction_error_scorer,
                #            'reward_error': dynamics_reward_prediction_error_scorer,
                #            'variance': dynamics_prediction_variance_scorer,
                #         },
                #         pretrain=False
                #     )

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
                    if self._use_mb_generate:
                        new_transitions, _ = self.generate_replay_data(
                            task_id,
                            dataset,
                            original,
                            in_task=True,
                            real_action_size=real_action_size,
                            real_observation_size=real_observation_size,
                        )
                    else:
                        new_transitions = None
                        # new_transitions = self.generate_new_data(
                        #     iterator.transitions,
                        #     real_observation_size=real_observation_size,
                        #     task_id=task_id,
                        # )

                    if new_transitions:
                        iterator.add_generated_transitions(new_transitions)
                        LOG.debug(
                            f"{len(new_transitions)} transitions are generated.",
                            real_transitions=len(iterator.transitions),
                            fake_transitions=len(iterator.generated_transitions),
                        )

                    # if new_transitions:
                    #     print(f'real_transitions: {len(iterator.transitions)}')
                    #     print(f'fake_transitions: {len(iterator.generated_transitions)}')
                    #     for new_transition in new_transitions:
                    #         mu, logstd = self._impl._policy.sample_with_log_prob(torch.from_numpy(new_transition.observation).to(self._impl.device))
                    #         print(f'mu: {mu}')
                    #         print(f'logstd: {logstd}')

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

                if epoch % save_interval == 0:
                    if scorers and eval_episodess:
                        for id, eval_episodes in eval_episodess.items():
                            scorers_tmp = {k + str(id): v(id, epoch) for k, v in scorers.items()}
                            self._evaluate(eval_episodes, scorers_tmp, logger)

                    if replay_scorers:
                        if replay_dataloaders is not None:
                            for replay_num, replay_dataloader in replay_dataloaders.items():
                                # 重命名
                                replay_scorers_tmp = {k + str(replay_num): v for k, v in replay_scorers.items()}
                                self._evaluate(replay_dataloader, replay_scorers_tmp, logger)

                # save metrics
                metrics = logger.commit(epoch, total_step)

                # save model parameters
                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                yield epoch, metrics

            # drop reference to active logger since out of fit there is no active
            # logger
            self._active_logger = None
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
                    if scorers and eval_episodess:
                        for id, eval_episodes in eval_episodess.items():
                            scorers_tmp = {k + str(id): v(id) for k, v in scorers.items()}
                            self._evaluate(eval_episodes, scorers_tmp, logger)

                    if replay_scorers:
                        if replay_dataloaders is not None:
                            for replay_num, replay_dataloader in replay_dataloaders.items():
                                # 重命名
                                replay_scorers_tmp = {k + str(replay_num): v for k, v in replay_scorers.items()}
                                self._evaluate(replay_dataloader, replay_scorers_tmp, logger)



        # # TODO: 这些还没写，别用！
        # # initialize scaler
        # if self._scaler:
        #     assert not self._scaler
        #     LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
        #     self._scaler.fit(episodes)
        #     if replay_episodess is not None:
        #         for replay_episodes in replay_episodess:
        #             self._scaler.fit(replay_episodes)

        # # initialize action scaler
        # if self._action_scaler:
        #     assert not self._action_scaler
        #     LOG.debug(
        #         "Fitting action scaler...",
        #         action_scaler=self._action_scaler.get_type(),
        #     )
        #     self._action_scaler.fit(episodes)
        #     if replay_episodess is not None:
        #         for replay_episodes in replay_episodess:
        #             self._action_scaler.fit(replay_episodes)

        # # initialize reward scaler
        # if self._reward_scaler:
        #     assert not self._reward_scaler
        #     LOG.debug(
        #         "Fitting reward scaler...",
        #         reward_scaler=self._reward_scaler.get_type(),
        #     )
        #     self._reward_scaler.fit(episodes)
        #     if replay_episodess is not None:
        #         for replay_episodes in replay_episodess:
        #             self._reward_scaler.fit(replay_episodes)

        # instantiate implementation

        # training loop

    def test(
        self,
        replay_datasets: Optional[Dict[int, List[TensorDataset]]],
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodess: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], Callable[..., float]]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, Iterator], float]]
        ] = None,
    ) -> List[Tuple[int, Dict[int, float]]]:
        results = list(
            self.tester(
                replay_datasets,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodess,
                save_interval,
                scorers,
                replay_scorers,
            )
        )
        return results

    def tester(
        self,
        replay_datasets: Optional[Dict[int, List[TensorDataset]]],
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodess: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], Callable[..., float]]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, Iterator], float]]
        ] = None,
    ) -> Generator[Tuple[int, Dict[int, float]], None, None]:

        replay_dataloaders: Optional[Dict[int, List[TensorDataset]]]
        if replay_datasets is not None:
            replay_dataloaders = dict()
            for replay_num, replay_dataset in replay_datasets.items():
                dataloader = DataLoader(replay_dataset, batch_size=self._batch_size, shuffle=True)
                replay_dataloaders[replay_num] = dataloader
            replay_iterators = dict()
            for replay_num, replay_dataloader in replay_dataloaders.items():
                replay_iterators[replay_num] = iter(replay_dataloader)
        else:
            replay_dataloaders = None
            replay_iterators = None

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

        # # TODO: 这些还没写，别用！
        # # initialize scaler
        # if self._scaler:
        #     assert not self._scaler
        #     LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
        #     self._scaler.fit(episodes)
        #     if replay_episodess is not None:
        #         for replay_episodes in replay_episodess:
        #             self._scaler.fit(replay_episodes)

        # # initialize action scaler
        # if self._action_scaler:
        #     assert not self._action_scaler
        #     LOG.debug(
        #         "Fitting action scaler...",
        #         action_scaler=self._action_scaler.get_type(),
        #     )
        #     self._action_scaler.fit(episodes)
        #     if replay_episodess is not None:
        #         for replay_episodes in replay_episodess:
        #             self._action_scaler.fit(replay_episodes)

        # # initialize reward scaler
        # if self._reward_scaler:
        #     assert not self._reward_scaler
        #     LOG.debug(
        #         "Fitting reward scaler...",
        #         reward_scaler=self._reward_scaler.get_type(),
        #     )
        #     self._reward_scaler.fit(episodes)
        #     if replay_episodess is not None:
        #         for replay_episodes in replay_episodess:
        #             self._reward_scaler.fit(replay_episodes)

        # instantiate implementation
        assert self._impl is not None
        self._impl.update_alpha()
        LOG.warning("Skip building models since they're already built.")

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

        total_step = 0
        epoch = 1

        # dict to add incremental mean losses to epoch
        epoch_loss = defaultdict(list)

        if replay_dataloaders is not None:
            replay_iterators = dict()
            for replay_num, replay_dataloader in replay_dataloaders.items():
                replay_iterators[replay_num] = iter(replay_dataloader)
        else:
            replay_iterators = None

        with logger.measure_time("step"):
            # pick transitions
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

        if scorers and eval_episodess:
            for id, eval_episodes in eval_episodess.items():
                scorers_tmp = {k + str(id): v(id) for k, v in scorers.items()}
                self._evaluate(eval_episodes, scorers_tmp, logger)

        if replay_scorers:
            if replay_dataloaders is not None:
                for replay_num, replay_dataloader in replay_dataloaders.items():
                    # 重命名
                    replay_scorers_tmp = {k + str(replay_num): v for k, v in replay_scorers.items()}
                    self._evaluate(replay_dataloader, replay_scorers_tmp, logger)

        # save metrics
        metrics = logger.commit(epoch, total_step)

        # save model parameters
        if epoch % save_interval == 0:
            logger.save_model(total_step, self)

        yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger = None

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, task_id
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

        task_id_tensor = np.eye(self._id_size)[task_id].squeeze()
        task_id_tensor = np.broadcast_to(task_id_tensor, (observations.shape[0], self._id_size))
        # task_id_tensor = torch.from_numpy(np.broadcast_to(task_id_tensor, (observations.shape[0], self._id_size))).to(torch.float32).to(self._impl.device)

        for _ in range(self._get_rollout_horizon()):

            # predict next state
            pred = self._dynamics.predict(observations[:, :real_observation_size], actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, next_rewards, variances = pred

            # regularize by uncertainty
            next_observations, next_rewards = self._mutate_transition(
                next_observations, next_rewards, variances
            )
            next_observations = np.concatenate([next_observations, task_id_tensor], axis=1)

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
                    next_action=next_actions[i],
                    next_reward=float(next_rewards[i][0]),
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

    def generate_new_data_trajectory(self, task_id, dataset, original_observation, in_task=False, max_export_time = 100, max_reward=None, real_action_size=1, real_observation_size=1, len_iterator=1):

        if not self._is_generating_new_data():
            return None
        # 关键算法
        _original = torch.from_numpy(original_observation).to(self._impl.device)
        task_id_tensor = np.eye(self._id_size)[task_id].squeeze()
        task_id_tensor = torch.from_numpy(np.broadcast_to(task_id_tensor, (_original.shape[0], self._id_size))).to(torch.float32).to(self._impl.device)
        original_observation = torch.cat([_original, task_id_tensor], dim=1)
        original_action = self._impl._policy(original_observation)
        replay_indexes = None
        new_transitions = []

        export_time = 0
        start_indexes = np.zeros(0)
        while (start_indexes.shape[0] != 0 or original_observation is not None) and export_time < max_export_time:
            if original_observation is not None:
                start_observations = original_observation
                start_actions = original_action
                start_rewards = [0]
                original_observation = None
            else:
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
            pred_observations = torch.cat([pred[:, :-1], task_id_tensor.expand(pred.shape[0], -1)], dim=1)
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
                    next_action = next_action[i].cpu().detach().numpy(),
                    next_reward = next_reward[i].cpu().detach().numpy(),
                    terminal = 0,
                )
                new_transitions.append(transition)

            if start_indexes.shape[0] > 0:
                line_indexes = dataset._actions[start_indexes[0], real_action_size:].astype(np.int64)
                near_indexes, _, _ = similar_mb(mus[0], logstds[0], dataset._observations[line_indexes, :real_observation_size], np.expand_dims(dataset._rewards, axis=1), self._dynamics._impl._dynamics, topk=self._topk, input_indexes=line_indexes)
            else:
                near_indexes, _, _ = similar_mb(mus[0], logstds[0], dataset._observations[:, :real_observation_size], np.expand_dims(dataset._rewards, axis=1), self._dynamics._impl._dynamics, topk=self._topk)
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

        new_transitions = self.generate_new_data(transitions=new_transitions, real_observation_size=real_observation_size, task_id=task_id, len_iterator=len_iterator)
        if self._td3_loss or in_task:
            return new_transitions
        elif self._policy_bc_loss and not in_task:
            replay_observations = torch.cat([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0)
            replay_actions = torch.cat([torch.from_numpy(transition.action) for transition in new_transitions], dim=0)
            replay_rewards = torch.cat([torch.from_numpy(transition.reward) for transition in new_transitions], dim=0)
            replay_next_observations = torch.cat([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0)
            replay_next_actions = torch.cat([torch.from_numpy(transition.next_action) for transition in new_transitions], dim=0)
            replay_next_rewards = torch.cat([torch.from_numpy(transition.next_reward) for transition in new_transitions], dim=0)
            replay_terminals = torch.cat([torch.from_numpy(transition.terminals) for transition in new_transitions], dim=0)
            replay_dists = self._impl.policy.dist(replay_observations)
            replay_means, replay_std_logs = replay_dists.mean, replay_dists.stddev
            replay_qs = self._impl.q_func(replay_observations, replay_actions)
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals, replay_means, replay_std_logs, replay_qs)
            return replay_dataset

    def generate_replay_data(self, task_id, dataset, original_observation, in_task=False, max_save_num=1000, max_export_time = 100, max_reward=None, real_action_size=1, real_observation_size=1):
        # 关键算法
        _original = torch.from_numpy(original_observation).to(self._impl.device)
        task_id_tensor = np.eye(self._id_size)[task_id].squeeze()
        task_id_tensor = torch.from_numpy(np.broadcast_to(task_id_tensor, (_original.shape[0], self._id_size))).to(torch.float32).to(self._impl.device)
        original_observation = torch.cat([_original, task_id_tensor], dim=1)
        original_action = self._impl._policy(original_observation)
        replay_indexes = None
        new_transitions = []

        export_time = 0
        start_indexes = np.zeros(0)
        while (start_indexes.shape[0] != 0 or original_observation is not None) and export_time < max_export_time and len(new_transitions) < max_save_num:
            if original_observation is not None:
                start_observations = original_observation
                start_actions = original_action
                original_observation = None
            else:
                start_observations = torch.from_numpy(dataset._observations[start_indexes]).to(self._impl.device)
                start_actions = self._impl._policy(start_observations)

            mus, logstds = [], []
            for model in self._dynamics._impl._dynamics._models:
                mu, logstd = model.compute_stats(start_observations[:, :real_observation_size], start_actions)
                mus.append(mu)
                logstds.append(logstd)
            mus = torch.stack(mus, dim=1)
            logstds = torch.stack(logstds, dim=1)
            mus = mus[torch.arange(start_observations.shape[0]), torch.randint(len(self._dynamics._impl._dynamics._models), size=(start_observations.shape[0],))]
            logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(self._dynamics._impl._dynamics._models), size=(start_observations.shape[0],))]

            near_indexes_list = []
            if start_indexes.shape[0] > 0:
                for i in start_indexes:
                    line_indexes = dataset._actions[start_indexes[i], real_action_size:].astype(np.int64)
                    near_indexes, _, _ = similar_mb(mus[i], logstds[i], dataset._observations[line_indexes, :real_observation_size], np.expand_dims(dataset._rewards, axis=1), self._dynamics._impl._dynamics, topk=self._topk, input_indexes=line_indexes)
                    near_indexes_list.append(near_indexes)
            else:
                near_indexes, _, _ = similar_mb(mus[0], logstds[0], dataset._observations[:, :real_observation_size], np.expand_dims(dataset._rewards, axis=1), self._dynamics._impl._dynamics, topk=self._topk)
                near_indexes_list.append(near_indexes)
            # 第一个将作为接下来衍生的起点被保留。
            near_indexes_list.reverse()
            for start_indexes in near_indexes_list:
                if replay_indexes is not None:
                    start_indexes = np.setdiff1d(start_indexes, replay_indexes, True)
                start_next_indexes = np.where(start_indexes + 1 < dataset._observations.shape[0], start_indexes + 1, 0)

                for i in range(start_indexes.shape[0]):
                    transition = Transition(
                        observation_shape = self._impl.observation_shape,
                        action_size = self._impl.action_size,
                        observation = dataset._observations[start_indexes[i]],
                        action = dataset._actions[start_indexes[i]][:real_action_size],
                        reward = dataset._rewards[start_indexes[i]],
                        next_observation = dataset._observations[start_next_indexes[i]],
                        next_action = dataset._actions[start_next_indexes[i]][:real_action_size],
                        next_reward = dataset._rewards[start_next_indexes[i]],
                        terminal = dataset._terminals[start_indexes[i]]
                    )
                    new_transitions.append(transition)

            start_rewards = dataset._rewards[start_indexes]
            if max_reward is not None:
                start_indexes = start_indexes[start_rewards >= max_reward]
            if start_indexes.shape[0] == 0:
                break
            if replay_indexes is not None:
                replay_indexes = np.concatenate([replay_indexes, start_indexes], axis=0)
            else:
                replay_indexes = start_indexes

        if len(new_transitions) > max_save_num:
            new_transitions = new_transitions[:max_save_num]

        replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0)
        replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in new_transitions], dim=0)
        replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in new_transitions], dim=0)
        replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0)
        replay_next_actions = torch.stack([torch.from_numpy(transition.next_action) for transition in new_transitions], dim=0)
        replay_next_rewards = torch.stack([torch.from_numpy(np.array([transition.next_reward])) for transition in new_transitions], dim=0)
        replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in new_transitions], dim=0)
        if self._td3_loss or in_task:
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals)
            return new_transitions, replay_dataset
        elif self._policy_bc_loss and not in_task:
            replay_dists = self._impl.policy.dist(replay_observations)
            replay_means, replay_std_logs = replay_dists.mean, replay_dists.stddev
            replay_qs = self._impl.q_func(replay_observations, replay_actions)
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals, replay_means, replay_std_logs, replay_qs)
            return replay_dataset, replay_dataset

    def generate_replay_data_random(self, task_id, dataset, in_task=False, max_save_num=1000, real_action_size=1):
        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset
        iterator = RandomIterator(
            episodes,
            max_save_num,
            batch_size=self._batch_size,
            n_steps=self._n_steps,
            gamma=self._gamma,
            n_frames=self._n_frames,
            real_ratio=self._real_ratio,
            generated_maxlen=self._generated_maxlen,
        )
        transitions = iterator.sample()
        task_id_numpy = np.eye(task_id)[self._id_size].squeeze()
        task_id_numpy = np.broadcast_to(task_id_numpy, (max_save_num, 1))
        new_transitions = []
        for transition in transitions:
            new_transitions.append(
                Transition(
                    observation_shape = transition.observation_shape,
                    action_size = transition.action_size,
                    observation = np.concatenate([transition.observation, task_id_numpy], axis=1),
                    action = transition.action[:, :real_action_size],
                    reward = transition.reward,
                    next_action = transition.next_action[:, :real_action_size],
                    next_reward = transition.next_reward,
                    terminal = transition.terminal,
                )
            )

            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in new_transitions], dim=0)
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in new_transitions], dim=0)
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in new_transitions], dim=0)
            replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in new_transitions], dim=0)
            replay_next_actions = torch.stack([torch.from_numpy(transition.next_action) for transition in new_transitions], dim=0)
            replay_next_rewards = torch.stack([torch.from_numpy(np.array([transition.next_reward])) for transition in new_transitions], dim=0)
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in new_transitions], dim=0)
        if self._td3_loss or in_task:
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals)
            return new_transitions, replay_dataset
        elif self._policy_bc_loss and not in_task:
            replay_dists = self._impl.policy.dist(replay_observations)
            replay_means, replay_std_logs = replay_dists.mean, replay_dists.stddev
            replay_qs = self._impl.q_func(replay_observations, replay_actions)
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals, replay_means, replay_std_logs, replay_qs)
            return replay_dataset, replay_dataset
