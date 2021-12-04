from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
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
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.dqn import DoubleDQN

from myd3rlpy.algos.torch.cqlbc_impl import CQLBCImpl


class CQLBC(AlgoBase):
    r"""Conservative Q-Learning algorithm.
    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.
    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.
    .. math::
        L(\theta_i) = \alpha\, \mathbb{E}_{s_t \sim D}
            \left[\log{\sum_a \exp{Q_{\theta_i}(s_t, a)}}
             - \mathbb{E}_{a \sim D} \big[Q_{\theta_i}(s_t, a)\big] - \tau\right]
            + L_\mathrm{SAC}(\theta_i)
    where :math:`\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\tau`, the
    :math:`\alpha` will become smaller.
    Otherwise, the :math:`\alpha` will become larger to aggressively penalize
    action-values.
    In continuous control, :math:`\log{\sum_a \exp{Q(s, a)}}` is computed as
    follows.
    .. math::
        \log{\sum_a \exp{Q(s, a)}} \approx \log{\left(
            \frac{1}{2N} \sum_{a_i \sim \text{Unif}(a)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\text{Unif}(a)}\right]
            + \frac{1}{2N} \sum_{a_i \sim \pi_\phi(a|s)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\pi_\phi(a_i|s)}\right]\right)}
    where :math:`N` is the number of sampled actions.
    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.
    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_
    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float):
            learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for :math:`\alpha`.
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
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\tau`.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.cql_impl.CQLImpl): algorithm implementation.
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
    _tau: float
    _n_critics: int
    _target_reduction_type: str
    _initial_temperature: float
    _initial_alpha: float
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    _use_gpu: Optional[Device]
    _impl: Optional[CQLBCImpl]

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
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        target_reduction_type: str = "min",
        initial_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 5.0,
        n_action_samples: int = 10,
        soft_q_backup: bool = False,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[CQLBCImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._alpha_learning_rate = alpha_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._alpha_optim_factory = alpha_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._target_reduction_type = target_reduction_type
        self._initial_temperature = initial_temperature
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

        self._replay_grad_step = 0

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CQLBCImpl(
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
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            target_reduction_type=self._target_reduction_type,
            initial_temperature=self._initial_temperature,
            initial_alpha=self._initial_alpha,
            alpha_threshold=self._alpha_threshold,
            conservative_weight=self._conservative_weight,
            n_action_samples=self._n_action_samples,
            soft_q_backup=self._soft_q_backup,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # lagrangian parameter update for SAC temperature
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        # lagrangian parameter update for conservative loss weight
        if self._alpha_learning_rate > 0:
            alpha_loss, alpha = self._impl.update_alpha(batch)
            metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def _replay_update(self, batch: List[Tensor]) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        critic_loss = self._impl.replay_update_critic(batch)
        metrics.update({"replay_critic_loss": critic_loss})

        actor_loss = self._impl.replay_update_actor(batch)
        metrics.update({"replay_actor_loss": actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def fit(
        self,
        dataset: Union[List[Episode], MDPDataset],
        replay_datasets: Optional[List[Union[Dataset]],
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
        replay_eval_episodess: Optional[List[List[ReplayEpisode]]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        replay_scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
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
                save_interval,
                scorers,
                shuffle,
                callback,
            )
        )
        return results

    def fitter(
        self,
        dataset: Union[List[Episode], MDPDataset],
        replay_datasets: Optional[List[Dataset]],
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
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
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
        replay_episodess is replay_datasets

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
        replay_dataloaders: Optional[List[Dataloader]]
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
            if replay_episodess is not None:
                replay_iterators = []
                for replay_episodes in replay_episodess:
                    dataloader = Dataloader(replay_episodes, batch_size=self._batch_size, shuffle=True)
                    # iterator = RandomIterator(
                    #     replay_episodes,
                    #     n_steps_per_epoch,
                    #     batch_size=self._batch_size,
                    #     n_steps=self._n_steps,
                    #     gamma=self._gamma,
                    #     n_frames=self._n_frames,
                    #     real_ratio=self._real_ratio,
                    #     generated_maxlen=self._generated_maxlen,
                    # )
                    replay_dataloaders.append(dataloader)
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
            if replay_episodess is not None:
                replay_dataloaders = []
                for replay_episodes in replay_episodess:
                    dataloader = Dataloader(replay_episodes, batch_size=self._batch_size, shuffle=True)
                    # iterator = RoundIterator(
                    #     replay_episodes,
                    #     batch_size=self._batch_size,
                    #     n_steps=self._n_steps,
                    #     gamma=self._gamma,
                    #     n_frames=self._n_frames,
                    #     real_ratio=self._real_ratio,
                    #     generated_maxlen=self._generated_maxlen,
                    #     shuffle=shuffle,
                    # )
                    replay_dataloaders.append(dataloader)
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
            action_size = transition.get_action_size()
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
            self._replay_loss_histories = []
            for _replay_loss_history in self._replay_loss_histories:
                _replay_loss_history = defaultdict(list)

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)
            if replay_datasets is not None:
                replay_epoch_losses = []
                for replay_dataset in replay_datasets:
                    replay_epoch_losses.append(defaultdict(list))

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            iterator.reset()
            if replay_dataloaders is not None:
                replay_iterators = []
                for replay_dataloader in replay_dataloaders:
                    replay_iterators.append(replay_dataloader)
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

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

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

                if replay_iterators is not None:
                    for replay_num, replay_iterator in enumerate(replay_iterators):
                        with logger.measure_time("replay step"):
                            # pick transitions
                            with logger.measure_time("sample_batch"):
                                batch = next(replay_iterator)

                            # update parameters
                            with logger.measure_time("algorithm_update"):
                                loss = self.replay_update(batch)

                            # record metrics
                            for name, val in loss.items():
                                logger.add_metric(name + str(replay_num), val)
                                replay_epoch_losses[replay_num][name].append(val)

                            # update progress postfix with losses
                            if itr % 10 == 0:
                                mean_loss = {
                                    k: np.mean(v) for k, v in replay_epoch_losses[replay_num].items()
                                }

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
                for replay_num, replay_epoch_loss in enumerate(replay_epoch_losses):
                    for name, vals in replay_epoch_loss.items():
                        if vals:
                            self._replay_loss_histories[replay_num][name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            if replay_scorers and replay_eval_episodess:
                if replay_iterators is not None:
                    for replay_num, replay_eval_episodess in enumerate(replay_eval_episodess):
                        # 重命名
                        replay_scorers_tmp = {k + str(replay_num): v for k, v in replay_eval_scorers.item()}
                        self._evaluate(replay_eval_episodes_tmp, replay_scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger = None

    def replay_update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._replay_update(batch)
        self._replay_grad_step += 1
        return loss
