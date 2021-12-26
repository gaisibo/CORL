from typing import Any, Dict, Optional, Sequence, List, Union, Callable, Tuple, Generator, Iterator
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
from functools import partial
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, DataLoader

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
from d3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.td3_plus_bc import TD3PlusBC
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from d3rlpy.base import TransitionIterator, TransitionMiniBatch, LearnableBase
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.iterators.random_iterator import RandomIterator
from d3rlpy.iterators.round_iterator import RoundIterator
from d3rlpy.logger import LOG, D3RLPyLogger

class MI(TD3PlusBC):
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
    # phi和psi与actor和critic共用encoder，不需要另外训练。
    # actor必须被重放，没用选择。
    _replay_actor_alpha: float
    _replay_critic_alpha: float
    _replay_critic: bool
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _target_reduction_type: str
    _target_smoothing_sigma: float
    _target_smoothing_clip: float
    _update_actor_interval: int
    _use_gpu: Optional[Device]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        replay_actor_alpha = 2.5,  # from A Minimalist Approach to Offline Reinforcement Learning
        replay_critic_alpha = 1,
        replay_critic: bool = True,
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        target_reduction_type: str = "min",
        target_smoothing_sigma: float = 0.2,
        target_smoothing_clip: float = 0.5,
        n_sample_actions: int = 10,
        update_actor_interval: int = 2,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl = None,
        impl_name = 'mi',
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
            target_reduction_type = target_reduction_type,
            target_smoothing_sigma = target_smoothing_sigma,
            target_smoothing_clip = target_smoothing_clip,
            update_actor_interval = update_actor_interval,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
            impl = impl,
            kwargs = kwargs,
        )
        self._replay_critic = replay_critic
        self._n_sample_actions = n_sample_actions
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._impl_name = impl_name

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        assert self._impl_name in ['mi', 'gemmi', 'agemmi']
        if self._impl_name == 'mi':
            from myd3rlpy.algos.torch.mi_impl import MIImpl
        elif self._impl_name == 'gemmi':
            from myd3rlpy.algos.torch.gemmi_impl import GEMMIImpl as MIImple
        elif self._impl_name == 'agemmi':
            from myd3rlpy.algos.torch.agemmi_impl import AGEMMIImpl as MIImple
        self._impl = MIImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            replay_critic_alpha=self._replay_critic_alpha,
            replay_actor_alpha=self._replay_actor_alpha,
            replay_critic=self._replay_critic,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            target_reduction_type=self._target_reduction_type,
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            n_sample_actions = self._n_sample_actions,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]]) -> Dict[str, float]:
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
    def _update(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[Tensor]]]) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        critic_loss, critic_q_func_loss, critic_replay_loss = self._impl.update_critic(batch, replay_batches=replay_batches) * self._critic_alpha()
        metrics.update({"critic_loss": critic_loss})
        metrics.update({"critic_q_func_loss": critic_q_func_loss})
        metrics.update({"critic_replay_loss": critic_replay_loss})

        # delayed policy update
        if self._grad_step % self._update_actor_interval == 0:
            actor_loss, actor_policy_loss, actor_replay_loss = self._impl.update_actor(batch, replay_batches=replay_batches) * self._actor_alpha()
            metrics.update({"actor_loss": actor_loss})
            metrics.update({"actor_policy_loss": actor_policy_loss})
            metrics.update({"actor_replay_loss": actor_replay_loss})
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def _critic_alpha(self):
        return 1

    def _actor_alpha(self):
        return 1

    def fit(
        self,
        dataset: Union[List[Episode], MDPDataset],
        replay_datasets: Optional[Dict[int, List[TensorDataset]]],
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
        eval_episodes: Optional[List[Episode]] = None,
        replay_eval_episodess: Optional[Dict[int, Iterator]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, Iterator, int], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        real_action_size: Optional[int] = None,
        real_observation_size: Optional[int] = None,
    ) -> List[Tuple[int, Dict[str, float]]]:
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
                dataset,
                replay_datasets,
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
                eval_episodes,
                replay_eval_episodess,
                save_interval,
                scorers,
                replay_scorers,
                shuffle,
                callback,
                real_action_size,
                real_observation_size,
            )
        )
        return results

    def fitter(
        self,
        dataset: Union[List[Episode], MDPDataset],
        replay_datasets: Optional[Dict[int, List[TensorDataset]]],
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
        eval_episodes: Optional[List[Episode]] = None,
        replay_eval_episodess: Optional[Dict[int, Iterator]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, Iterator], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        real_action_size: Optional[int] = None,
        real_observation_size: Optional[int] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
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

        if isinstance(dataset, MDPDataset):
            episodes = dataset.episodes
        else:
            episodes = dataset

        # check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif len(episodes[0].actions.shape) > 1:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR

        iterator: TransitionIterator
        replay_dataloaders: Optional[Dict[int, List[TensorDataset]]]
        if n_epochs is None and n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = RandomIterator(
                episodes,
                n_steps_per_epoch,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
            if replay_datasets is not None:
                replay_dataloaders = dict()
                for replay_num, replay_dataset in replay_datasets.items():
                    dataloader = DataLoader(replay_dataset, batch_size=self._batch_size, shuffle=True)
                    replay_dataloaders[replay_num] = dataloader
            else:
                replay_dataloaders = None
            LOG.debug("RandomIterator is selected.")
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                episodes,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
            if replay_datasets is not None:
                replay_dataloaders = dict()
                for replay_num, replay_dataset in replay_datasets.items():
                    dataloader = DataLoader(replay_dataset, batch_size=self._batch_size, shuffle=True)
                    replay_dataloaders[replay_num] = dataloader
            else:
                replay_dataloaders = None
            LOG.debug("RoundIterator is selected.")
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")

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

        # TODO: 这些还没写，别用！
        # initialize scaler
        if self._scaler:
            assert not self._scaler
            LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
            self._scaler.fit(episodes)
            if replay_episodess is not None:
                for replay_episodes in replay_episodess:
                    self._scaler.fit(replay_episodes)

        # initialize action scaler
        if self._action_scaler:
            assert not self._action_scaler
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=self._action_scaler.get_type(),
            )
            self._action_scaler.fit(episodes)
            if replay_episodess is not None:
                for replay_episodes in replay_episodess:
                    self._action_scaler.fit(replay_episodes)

        # initialize reward scaler
        if self._reward_scaler:
            assert not self._reward_scaler
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=self._reward_scaler.get_type(),
            )
            self._reward_scaler.fit(episodes)
            if replay_episodess is not None:
                for replay_episodes in replay_episodess:
                    self._reward_scaler.fit(replay_episodes)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            transition = iterator.transitions[0]
            action_size = real_action_size
            observation_shape = real_observation_size
            observation_shape = tuple(transition.get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )
            LOG.debug("Models have been built.")
        else:
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

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)
            if replay_datasets is not None:
                replay_epoch_losses = dict()
                for replay_num, replay_dataset in replay_datasets.items():
                    replay_epoch_losses[replay_num] = defaultdict(list)
            else:
                replay_epoch_losses = None

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
                new_transitions = self.generate_new_data(
                    transitions=iterator.transitions,
                )
                if new_transitions:
                    iterator.add_generated_transitions(new_transitions)
                    LOG.debug(
                        f"{len(new_transitions)} transitions are generated.",
                        real_transitions=len(iterator.transitions),
                        fake_transitions=len(iterator.generated_transitions),
                    )

                # model based only, useless
                # if replay_iterators is not None:
                #     for replay_iterator in replay_iterators:
                #         self._scaler.fit(replay_iterator)
                #     replay_new_transitionss = self.generate_new_data(
                #         transitions=replay_iterator.transitions
                #     )
                #     LOG.debug(
                #         f"{len(replay_new_transitions)} replay_transitions are generated.",
                #         real_transitions=len(replay_iterator.transitions),
                #         fake_transitions=len(replay_iterator.generated_transitions),
                #     )

                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)
                        if replay_iterators is not None:
                            assert replay_dataloaders is not None
                            replay_batches = dict()
                            for replay_iterator_num in replay_iterators.items():
                                try:
                                    replay_batches[str(replay_iterator_num)] = next(replay_iterators[replay_iterator_num])
                                except StopIteration:
                                    replay_iterators[replay_iterator_num] = iter(replay_dataloaders[replay_iterator_num])
                                    replay_batches[str(replay_iterator_num)] = next(replay_iterators[replay_iterator_num])
                        else:
                            replay_batches = None

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch, replay_batches)

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
            if replay_iterators is not None:
                assert replay_epoch_losses is not None
                for replay_num, replay_epoch_loss in replay_epoch_losses.items():
                    for name, vals in replay_epoch_loss.items():
                        if vals:
                            self._replay_loss_histories[replay_num][name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            if replay_scorers and replay_eval_episodess:
                if replay_iterators is not None:
                    for replay_num, replay_eval_episodes in replay_eval_episodess.items():
                        # 重命名
                        replay_scorers_tmp = {k + str(replay_num): v for k, v in replay_scorers.items()}
                        self._replay_evaluate(replay_eval_episodes, replay_scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger = None
