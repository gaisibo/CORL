import tqdm
import numpy as np
from typing import Dict, Optional, List, Sequence, Any
from torch import Tensor
from d3rlpy.algos.base import AlgoBase

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
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory

from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR

from myd3rlpy.algos.o2o import O2OBase
from myd3rlpy.algos.torch.o2o_impl import O2OImpl

class O2OBPPO(O2OBase, AlgoBase):
    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _impl: Optional[O2OImpl]

    def __init__(
        self,
        *,
        actor_replay_type: float = 3e-4,
        critic_replay_type: float = 3e-4,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 1,
        update_critic_target_interval: int = 2,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[O2OBPPOImpl] = None,

        is_clip_decay = True,
        is_lr_decay = True,
        update_critic: bool = False,
        update_critic_interval: int = 10,
        const_eps: float = 1e-10,
        update_old_policy: bool = True,
        update_old_policy_interval: int = 10,

        clone_critic: bool = False,
        clone_actor: bool = False,
        **kwargs: Any
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
        self._actor_replay_type = actor_replay_type
        self._critic_replay_type = critic_replay_type
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_critic_target_interval = update_critic_target_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

        self._is_clip_decay = is_clip_decay
        self._is_lr_decay = is_lr_decay
        self._update_critic = update_critic
        self._update_critic_interval = update_critic_interval
        self._const_eps = const_eps
        self._update_old_policy = update_old_policy
        self._update_old_policy_interval = update_old_policy_interval

        self._clone_critic = clone_critic
        self._clone_actor = clone_actor

        self._value_grad_step = 0

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        impl_dict = {
            'observation_shape':observation_shape,
            'action_size':action_size,
            'actor_replay_type':self._actor_replay_type,
            'critic_replay_type':self._critic_replay_type,
            'actor_learning_rate':self._actor_learning_rate,
            'critic_learning_rate':self._critic_learning_rate,
            'actor_replay_lambda':self._actor_replay_lambda,
            'critic_replay_lambda':self._critic_replay_lambda,
            'temp_learning_rate':self._temp_learning_rate,
            'actor_optim_factory':self._actor_optim_factory,
            'critic_optim_factory':self._critic_optim_factory,
            'temp_optim_factory':self._temp_optim_factory,
            'actor_encoder_factory':self._actor_encoder_factory,
            'critic_encoder_factory':self._critic_encoder_factory,
            'q_func_factory':self._q_func_factory,
            'gamma':self._gamma,
            'gem_alpha':self._gem_alpha,
            'agem_alpha':self._agem_alpha,
            'ewc_rwalk_alpha':self._ewc_rwalk_alpha,
            'damping':self._damping,
            'epsilon':self._epsilon,
            'tau':self._tau,
            'n_critics':self._n_critics,
            'initial_temperature':self._initial_temperature,
            'use_gpu':self._use_gpu,
            'scaler':self._scaler,
            'action_scaler':self._action_scaler,
            'reward_scaler':self._reward_scaler,
            'fine_tuned_step': self._fine_tuned_step,

            'is_clip_decay': self._is_clip_decay,
            'is_lr_decay': self._is_lr_decay,
        }
        if self._impl_name == 'ppo':
            from myd3rlpy.algos.torch.o2o_ppo_impl import O2OPPOImpl as O2OImpl
        elif self._impl_name == 'bppo':
            from myd3rlpy.algos.torch.o2o_bppo_impl import O2OBPPOImpl as O2OImpl
            impl_dict["const_eps"] = self._const_eps
        else:
            print(self._impl_name)
            raise NotImplementedError
        self._impl = O2OImpl(
            **impl_dict
        )
        self._impl.build()

    def _update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        if self._grad_step > 200:
            self._impl._is_clip_decay = False
            self._impl._is_lr_decay = False
        if self._update_critic or online:
            critic_loss, replay_critic_loss = 0, 0
            for _ in range(self._update_critic_interval):
                critic_loss, replay_critic_loss = self._impl.update_critic(value_batch, None, clone_critic=self._clone_critic, online=online)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})
        actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, clone_actor=self._clone_actor, online=online)
        metrics.update({"actor_loss": actor_loss})
        metrics.update({"replay_actor_loss": replay_actor_loss})
        if self._update_old_policy:
            self._impl.set_old_policy()

        return metrics

    def update_value(self, value_batch: TransitionMiniBatch) -> Dict[int, float]:
    # def update(self, batch: TransitionMiniBatch, online: bool = False, batch_num: int=0, total_step: int=0, replay_batch: Optional[List[Tensor]]=None) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._update_value(value_batch)
        self._value_grad_step += 1
        return loss

    def _update_value(self, value_batch: TransitionMiniBatch) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        critic_loss, value_loss = self._impl.update_critic_clone(value_batch)
        metrics.update({"value_critic_loss": critic_loss})
        metrics.update({"value_value_loss": value_loss})
        if self._value_grad_step % self._update_critic_target_interval == 0:
            self._impl.update_critic_target()
        return metrics

    def update_bc(self, bc_batch: TransitionMiniBatch) -> Dict[int, float]:
    # def update(self, batch: TransitionMiniBatch, online: bool = False, batch_num: int=0, total_step: int=0, replay_batch: Optional[List[Tensor]]=None) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._update_bc(bc_batch)
        self._bc_grad_step += 1
        return loss

    def _update_bc(self, bc_batch: TransitionMiniBatch) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        bc_loss = self._impl.update_bc(bc_batch)
        metrics.update({"bc_loss": bc_loss})
        if self._bc_grad_step % self._update_critic_target_interval == 0:
            self._impl.update_critic_target()
        return metrics

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, real_action_size, batch_size = 64,
    ) -> Optional[List[Transition]]:
        return None

    def fitter(
        self,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        iterator: Optional[TransitionIterator] = None,
        value_iterator: Optional[TransitionIterator] = None,
        bc_iterator: Optional[TransitionIterator] = None,
        old_iterator: Optional[TransitionIterator] = None,
        old_value_iterator: Optional[TransitionIterator] = None,
        old_bc_iterator: Optional[TransitionIterator] = None,
        buffer_mix_ratio: float = 0.5,
        buffer_mix_type: str = "all",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        n_value_epochs: int = 2000,
        n_bc_epochs: int = 500,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        scorers_list: Optional[
                List[Dict[str, Callable[[Any, List[Episode]], float]]]
            ] = None,
        eval_episodes_list: Optional[Dict[int, List[Episode]]] = None,
        save_epochs: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        test: bool = False,
        epoch_num: Optional[int] = None,
    ):
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
            scorers: list of scorer functions used with `eval_episodes`.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            iterator yielding current epoch and metrics dict.
        """
        if self._impl is None:
            LOG.debug("Building models...")
            observation_shape = dataset.get_observation_shape()
            self.create_impl(
                self._process_observation_shape(observation_shape),
                dataset.get_action_size(),
            )
            self._impl._impl_id = 0
            LOG.debug("Models have been built.")
        else:
            self._impl._impl_id = 0
            # self._impl.rebuild_critic()
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

        total_step = 0
        print("train value and Q")
        for epoch in range(1, n_value_epochs + 1):
            if epoch > 1 and test:
                break
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(value_iterator)),
                disable=not show_progress,
                desc=f"Epoch {epoch}/{n_epochs}",
            )

            iterator.reset()

            for batch_num, itr in enumerate(range_gen):
                if batch_num > 10 and test:
                    break
                with logger.measure_time("step"):
                    with logger.measure_time("sample_batch"):
                        if old_value_iterator is not None:
                            new_batch = next(old_value_iterator)
                            if buffer_mix_type in ['all', 'value']:
                                try:
                                    old_batch = next(old_value_iterator)
                                except StopIteration:
                                    old_value_iterator.reset()
                                    old_batch = next(old_value_iterator)
                                part_new_batch = TransitionMiniBatch(new_batch.transitions[:round((1 - buffer_mix_ratio) * self._batch_size)])
                                mix_batch = TransitionMiniBatch(
                                        part_new_batch.transitions + old_batch.transitions
                                        )
                                value_batch = mix_batch
                            elif buffer_mix_type == 'policy':
                                value_batch = new_batch
                            else:
                                raise NotImplementedError
                        else:
                            value_batch = next(value_iterator)
                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update_value(value_batch)
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

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters

        print("train bc")
        for epoch in range(1, n_bc_epochs + 1):
            if epoch > 1 and test:
                break
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(bc_iterator)),
                disable=not show_progress,
                desc=f"Epoch {epoch}/{n_epochs}",
            )

            iterator.reset()

            for batch_num, itr in enumerate(range_gen):
                if batch_num > 10 and test:
                    break
                with logger.measure_time("step"):
                    with logger.measure_time("sample_batch"):
                        if old_iterator is not None:
                            new_batch = next(bc_iterator)
                            try:
                                old_batch = next(old_bc_iterator)
                            except StopIteration:
                                old_iterator.reset()
                                old_batch = next(old_bc_iterator)
                            part_new_batch = TransitionMiniBatch(new_batch.transitions[:round((1 - buffer_mix_ratio) * self._batch_size)])
                            mix_batch = TransitionMiniBatch(
                                    part_new_batch.transitions + old_batch.transitions
                                    )
                            if buffer_mix_type in ['all', 'policy']:
                                policy_batch = mix_batch
                            elif buffer_mix_type == 'value':
                                policy_batch = new_batch
                            else:
                                raise NotImplementedError
                        else:
                            policy_batch = next(iterator)
                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update_bc(policy_batch)
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

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters

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

            for batch_num, itr in enumerate(range_gen):
                if batch_num > 10 and test:
                    break

                # new_transitions = self.generate_new_data(transitions=iterator.transitions)
                # if new_transitions:
                #     iterator.add_generated_transitions(new_transitions)
                #     LOG.debug(
                #         f"{len(new_transitions)} transitions are generated.",
                #         real_transitions=len(iterator.transitions),
                #         fake_transitions=len(iterator.generated_transitions),
                #     )
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        if old_iterator is not None:
                            new_batch = next(iterator)
                            try:
                                old_batch = next(old_iterator)
                            except StopIteration:
                                old_iterator.reset()
                                old_batch = next(old_iterator)
                            part_new_batch = TransitionMiniBatch(new_batch.transitions[:round((1 - buffer_mix_ratio) * self._batch_size)])
                            mix_batch = TransitionMiniBatch(
                                    part_new_batch.transitions + old_batch.transitions
                                    )
                            if buffer_mix_type == 'all':
                                value_batch = policy_batch = mix_batch
                            elif buffer_mix_type == 'policy':
                                policy_batch = mix_batch
                                value_batch = new_batch
                            elif buffer_mix_type == 'value':
                                policy_batch = new_batch
                                value_batch = mix_batch
                            else:
                                raise NotImplementedError
                        else:
                            policy_batch = next(iterator)
                            value_batch = policy_batch

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(policy_batch, value_batch, online=False)
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

            if scorers_list and eval_episodes_list:
                for scorer_num, (scorers, eval_episodes) in enumerate(zip(scorers_list, eval_episodes_list)):
                    rename_scorers = dict()
                    for name, scorer in scorers.items():
                        rename_scorers[str(scorer_num) + '_' + name] = scorer
                    #print("test predict: {self._impl.predict_best_action()}")
                    self._evaluate(eval_episodes, rename_scorers, logger)
            else:
                print(f"scorers_list: {scorers_list}")
                print(f"eval_episodes_list: {eval_episodes_list}")
                assert False

            # save metrics
            logger.commit(epoch, total_step)

            if save_epochs is not None and save_path is not None and epoch in save_epochs:
                torch.save({'buffer': None, 'algo': self}, save_path.replace(str(n_epochs * n_steps_per_epoch), str(epoch * n_steps_per_epoch), 1))

            # save model parameters
        logger.close()

    def fit_online(
        self,
        env: gym.envs,
        eval_env: gym.envs,
        buffer: Optional[Buffer],
        old_buffer: Optional[Buffer] = None,
        buffer_mix_ratio: float = 0.5,
        buffer_mix_type: str = "all",
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        start_epoch: int = 0,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_step: int = 100000,
        eval_epsilon: float = 0.0,
        save_metrics: bool = True,
        save_steps: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        test: bool = False,
        # train_dynamics = False,
    ):
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
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            list of result tuples (epoch, metrics) per epoch.
        """
        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"

        logger = D3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            with_timestamp=with_timestamp,
        )

        if buffer is None:
            buffer = ReplayBuffer(n_steps, env)

        if self._impl is None:
            LOG.debug("Building models...")
            observation_shape = env.observation_space.shape
            self.create_impl(
                self._process_observation_shape(observation_shape),
                dataset.get_action_size(),
            )
            self._impl._impl_id = 0
            LOG.debug("Models have been built.")
        else:
            # self._impl.rebuild_critic()
            self._impl._impl_id = 0
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        ## setup evaluation scorer
        #eval_scorer: Optional[Callable[..., float]]
        #if eval_env:
        #    eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
        #else:
        #    eval_scorer = None
        # start training loop
        observation, _ = env.reset()
        if random_step > 0:
            exploit_observation, _ = eval_env.reset()
        rollout_return = 0.0


        for total_step in xrange(1, n_steps + 1):
            if total_step > 2000 and test:
                break
            with logger.measure_time("step"):
                #observation = observation.astype("f4")
                #fed_observation = observation

                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_step:
                        action = env.action_space.sample()
                        exploit_action = self.sample_action(observation[np.newaxis, :])
                        exploit_action = exploit_action[0]
                    else:
                        #action = self.sample_action([fed_observation])[0]
                        action = self.sample_action(observation[np.newaxis, :])
                        action = action[0]

                # step environment
                episode_length = 0
                with logger.measure_time("environment_step"):
                    if total_step < random_step:
                        exploit_next_observation, exploit_reward, exploit_terminal, exploit_truncated, exploit_info = eval_env.step(exploit_action)
                        rollout_return += exploit_reward
                    else:
                        next_observation, reward, terminal, truncated, info = env.step(action)
                        rollout_return += reward
                    episode_length += 1

                # special case for TimeLimit wrapper
                if truncated:
                    clip_episode = True
                    terminal = False
                    episode_length = 0
                else:
                    episode_length += 1
                    if episode_length == 1000 - 1:
                        terminal = True
                        episode_length = 0
                    clip_episode = terminal

                # store observation
                buffer.append(
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    clip_episode=clip_episode,
                )

                # reset if terminated
                if clip_episode:
                    observation, _ = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    rollout_return = 0.0
                    # for image observation
                else:
                    observation = next_observation

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch + start_epoch

                if total_step > update_start_step and len(buffer) > self._batch_size:
                    if total_step % update_interval == 0:
                        # sample mini-batch
                        with logger.measure_time("sample_batch"):
                            if old_buffer is not None:
                                new_batch = buffer.sample(
                                    batch_size=self._batch_size,#round((1 - buffer_mix_ratio) * self._batch_size),
                                    n_frames=self._n_frames,
                                    n_steps=self._n_steps,
                                    gamma=self._gamma,
                                )
                                part_new_batch = OldTransitionMiniBatch(new_batch.transitions[:round((1 - buffer_mix_ratio) * self._batch_size)])
                                old_batch = old_buffer.sample(
                                    batch_size=round(buffer_mix_ratio * self._batch_size),
                                    n_frames=self._n_frames,
                                    n_steps=self._n_steps,
                                    gamma=self._gamma,
                                )
                                mix_batch = OldTransitionMiniBatch(
                                        part_new_batch.transitions + old_batch.transitions
                                        )
                                if buffer_mix_type == 'all':
                                    value_batch = policy_batch = mix_batch
                                elif buffer_mix_type == 'policy':
                                    policy_batch = mix_batch
                                    value_batch = new_batch
                                elif buffer_mix_type == 'value':
                                    policy_batch = new_batch
                                    value_batch = mix_batch
                                else:
                                    raise NotImplementedError
                            else:
                                policy_batch = buffer.sample(
                                    batch_size=self._batch_size,
                                    n_frames=self._n_frames,
                                    n_steps=self._n_steps,
                                    gamma=self._gamma,
                                )
                                value_batch = policy_batch

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(policy_batch, value_batch, online=True)

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > start_epoch and total_step % n_steps_per_epoch == 0:
                # save metrics
                logger.commit(epoch, total_step)

            if save_steps is not None and save_path is not None and total_step in save_steps:
                buffer.clip_episode()
                torch.save({'buffer': buffer.to_mdp_dataset(), 'algo': self}, save_path)

        # clip the last episode
        buffer.clip_episode()

        # close logger
        logger.close()
