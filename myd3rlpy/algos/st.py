import copy
import time
import random
from typing import Any, Dict, Optional, List, Union, Callable, Tuple, Generator, Iterator, cast
from collections import defaultdict
from tqdm.auto import tqdm
from tqdm.auto import trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal

from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.online.buffers import Buffer, ReplayBuffer
from d3rlpy.torch_utility import _get_attributes
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR
from d3rlpy.base import LearnableBase
from d3rlpy.logger import LOG, D3RLPyLogger
import gym

from myd3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch, Transition
from d3rlpy.dataset import MDPDataset as OldMDPDataset
from myd3rlpy.iterators.base import TransitionIterator
from myd3rlpy.iterators.random_iterator import RandomIterator
from myd3rlpy.iterators.round_iterator import RoundIterator

# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STBase():
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
    _sample_type: str

    def update(self, batch: TransitionMiniBatch, online: bool = False, batch_num: int=0, total_step: int=0, replay_batch: TransitionMiniBatch=None) -> Dict[int, float]:
    # def update(self, batch: TransitionMiniBatch, online: bool = False, batch_num: int=0, total_step: int=0, replay_batch: Optional[List[Tensor]]=None) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._update(batch, online, batch_num, total_step, replay_batch)
        self._grad_step += 1
        return loss

    # 注意欧氏距离最近邻被塞到actions后面了。
    def _merge_update(self, batch: TransitionMiniBatch, replay_batch: TransitionMiniBatch=None) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        critic_loss = self._impl.merge_update_critic(batch, replay_batch)
        metrics.update({"critic_loss": critic_loss})

        actor_loss = self._impl.merge_update_actor(batch, replay_batch)
        metrics.update({"actor_loss": actor_loss})

        return metrics

    def build_with_dataset(self, dataset, dataset_id):
        if self._impl is None:
            LOG.debug("Building models...")
            observation_shape = dataset.get_observation_shape()
            self.create_impl(
                self._process_observation_shape(observation_shape),
                dataset.get_action_size(),
            )
            self._impl._impl_id = dataset_id
            LOG.debug("Models have been built.")
        else:
            self._impl._impl_id = dataset_id
            # self._impl.rebuild_critic()
            LOG.warning("Skip building models since they're already built.")

    def make_iterator(self, dataset, replay_dataset, n_steps, n_steps_per_epoch, n_epochs, shuffle):
        iterator: TransitionIterator
        assert dataset is not None

        transitions = self.make_transitions(dataset)

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

        if replay_dataset is not None:
            replay_iterator: TransitionIterator
            assert dataset is not None
            replay_transitions = []
            if isinstance(dataset, MDPDataset):
                for episode in cast(MDPDataset, replay_dataset).episodes:
                    replay_transitions += episode.transitions
            elif not dataset:
                raise ValueError("empty dataset is not supported.")
            elif isinstance(replay_dataset[0], Episode):
                for replay_episode in cast(List[Episode], replay_dataset):
                    replay_transitions += replay_episode.transitions
            elif isinstance(replay_dataset[0], Transition):
                replay_transitions = list(cast(List[Transition], replay_dataset))
            else:
                raise ValueError(f"invalid dataset type: {type(replay_dataset)}")

            if n_steps is not None:
                assert n_steps >= n_steps_per_epoch
                n_epochs = n_steps // n_steps_per_epoch
                replay_iterator = RandomIterator(
                    replay_transitions,
                    n_steps_per_epoch,
                    batch_size=self._batch_size,
                    n_steps=self._n_steps,
                    gamma=self._gamma,
                    n_frames=self._n_frames,
                    real_ratio=self._real_ratio,
                    generated_maxlen=self._generated_maxlen,
                )
            elif n_epochs is not None and n_steps is None:
                replay_iterator = RoundIterator(
                    replay_transitions,
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
        else:
            replay_iterator = None
        return iterator, replay_iterator, n_epochs

    def make_transitions(self, dataset):
        transitions = []
        if isinstance(dataset, MDPDataset):
            for episode in cast(MDPDataset, dataset).episodes:
                transitions += episode.transitions
        elif isinstance(dataset, OldMDPDataset):
            for episode in cast(OldMDPDataset, dataset).episodes:
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
        return transitions

    def fit(
        self,
        dataset_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        iterator: Optional[TransitionIterator] = None,
        replay_dataset: Optional[Union[TensorDataset, List[Transition]]] = None,
        replay_iterator: Optional[Iterator] = None,
        n_epochs: Optional[int] = None,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        eval_episodes_list: Optional[List[List[Episode]]] = None,
        save_interval: int = 10,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 1,
        eval_freq: int = 50,
        scorers_list: Optional[
                List[Dict[str, Callable[[Any, List[Episode]], float]]]
            ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        test: bool = False,
        epoch_num: Optional[int] = None,
        # train_dynamics = False,
    ) -> List[Tuple[int, float]]:
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
        result = list(
            self.fitter(
                dataset_id,
                dataset,
                iterator,
                replay_dataset,
                replay_iterator,
                n_epochs,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodes_list,
                save_interval,
                discount,
                start_timesteps,
                expl_noise,
                eval_freq,
                scorers_list,
                shuffle,
                callback,
                test,
                epoch_num,
                # train_dynamics,
            )
        )
        return result

    def fitter(
        self,
        dataset_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        iterator: Optional[TransitionIterator] = None,
        replay_dataset: Optional[Union[TensorDataset, List[Transition]]] = None,
        replay_iterator: Optional[TransitionIterator] = None,
        n_epochs: Optional[int] = None,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        eval_episodes_list: Optional[Dict[int, List[Episode]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
	scorers_list: Optional[
            List[Dict[str, Callable[[Any, List[Episode]], float]]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        test: bool = False,
        epoch_num: Optional[int] = None,
    ) -> Generator[Tuple[int, float], None, None]:
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
            observation_shape = dataset.get_observation_shape()
            self.create_impl(
                self._process_observation_shape(observation_shape),
                dataset.get_action_size(),
            )
            self._impl._impl_id = dataset_id
            LOG.debug("Models have been built.")
        else:
            self._impl._impl_id = dataset_id
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
            if replay_iterator is not None:
                replay_iterator.reset()

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
                        batch = next(iterator)
                        if replay_iterator is not None:
                            replay_batch = dict()
                            try:
                                replay_batch = next(replay_iterator)
                            except StopIteration:
                                replay_iterator.reset()
                                replay_batch = next(replay_iterator)
                        else:
                            replay_batch = None

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch, batch_num=batch_num, total_step=total_step, replay_batch=replay_batch)
                        # self._impl.increase_siamese_alpha(epoch - n_epochs, itr / len(iterator))

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    try:
                        # logger.add_metric("weight", self._impl._weight)
                        # logger.add_metric("log_probs", self._impl._log_probs)
                        # logger.add_metric("q_t", self._impl._q_t)
                        # logger.add_metric("v_t", self._impl._v_t)
                        logger.add_metric("q_loss", self._impl._q_loss.detach().cpu().numpy())
                        logger.add_metric("v_loss", self._impl._v_loss.detach().cpu().numpy())
                        epoch_loss["q_loss"].append(self._impl._q_loss.detach().cpu().numpy())
                        epoch_loss["v_loss"].append(self._impl._v_loss.detach().cpu().numpy())
                        logger.add_metric("replay_q_loss", self._impl._replay_q_loss.detach().cpu().numpy())
                        logger.add_metric("replay_v_loss", self._impl._replay_v_loss.detach().cpu().numpy())
                        epoch_loss["replay_q_loss"].append(self._impl._replay_q_loss.detach().cpu().numpy())
                        epoch_loss["replay_v_loss"].append(self._impl._replay_v_loss.detach().cpu().numpy())
                        # logger.add_metric("replay_weight", self._impl._replay_weight)
                        # logger.add_metric("replay_log_probs", self._impl._replay_log_probs)
                        # logger.add_metric("replay_q_t", self._impl._replay_q_t)
                        # logger.add_metric("replay_v_t", self._impl._replay_v_t)
                        # logger.add_metric("replay_q_loss", self._impl._replay_q_loss)
                        # logger.add_metric("replay_v_loss", self._impl._replay_v_loss)
                    except AttributeError:
                        pass

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
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

    #def after_learn(self, iterator, experiment_name, scorers_list, eval_episodes_list, logdir='d3rlpy_logs'):
    def after_learn(self, iterator):
        # for EWC
        if self._critic_replay_type in ['rwalk', 'ewc']:
            self._impl.critic_ewc_rwalk_post_train_process(iterator)
        # elif self._critic_replay_type == 'si':
        #     self._impl.critic_si_post_train_process()
        # elif self._critic_replay_type == 'gem':
        #     self._impl.gem_post_train_process()
        if self._actor_replay_type in ['rwalk', 'ewc']:
            self._impl.actor_ewc_rwalk_post_train_process(iterator)
        # elif self._actor_replay_type == 'si':
        #     self._impl.actor_si_post_train_process()
        #if self._impl_name in ['mgcql', 'mqcql', 'mrcql']:
        #    self._impl.match_prop_post_train_process(iterator)
        # self._impl.reinit_network()

        # # TEST
        # for scorer_num, (scorers, eval_episodes) in enumerate(zip(scorers_list, eval_episodes_list)):
        #     # setup logger
        #     logger = self._prepare_logger(True, experiment_name, True, logdir, True, None)
        #     rename_scorers = dict()
        #     for name, scorer in scorers.items():
        #         rename_scorers[str(scorer_num) + '_' + name] = scorer
        #     self._evaluate(eval_episodes, rename_scorers, logger)
        #     # save metrics
        #     metrics = logger.commit(0, 0)

        if self._clone_actor:
            self._impl.save_clone_policy()
            # self._impl.save_clone_data()
        self._impl._targ_q_func = copy.deepcopy(self._impl._q_func)
        self._impl._targ_policy = copy.deepcopy(self._impl._policy)

    def score(self, scorers_list, eval_episodes_list, iterator, experiment_name, logdir='d3rlpy_logs'):
        if scorers_list and eval_episodes_list:
            for scorer_num, (scorers, eval_episodes) in enumerate(zip(scorers_list, eval_episodes_list)):
                # setup logger
                logger = self._prepare_logger(True, experiment_name, True, logdir, True, None)
                rename_scorers = dict()
                for name, scorer in scorers.items():
                    rename_scorers[str(scorer_num) + '_' + name] = scorer
                self._evaluate(eval_episodes, rename_scorers, logger)
                metrics = logger.commit(0, 0)

    def fit_online(
        self,
        env: gym.envs,
        eval_env: gym.envs,
        buffer: Optional[Buffer],
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        start_epoch: int = 0,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_step: int = 100000,
        eval_epsilon: float = 0.0,
        save_metrics: bool = True,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        timelimit_aware: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        test: bool = False,
        # train_dynamics = False,
    ) -> List[Tuple[int, float]]:
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

        # setup evaluation scorer
        eval_scorer: Optional[Callable[..., float]]
        if eval_env:
            eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
        else:
            eval_scorer = None
        # start training loop
        observation, _ = env.reset()
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
                    else:
                        #action = self.sample_action([fed_observation])[0]
                        action = self.sample_action(observation[np.newaxis, :])
                        action = action[0]

                # step environment
                episode_length = 0
                with logger.measure_time("environment_step"):
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
                            batch = buffer.sample(
                                batch_size=self._batch_size,
                                n_frames=self._n_frames,
                                n_steps=self._n_steps,
                                gamma=self._gamma,
                            )

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(batch, online=True)

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > start_epoch and total_step % n_steps_per_epoch == 0:
                # evaluation
                #if eval_scorer:
                #    logger.add_metric("evaluation", np.mean(reward))

                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                # save metrics
                logger.commit(epoch, total_step)

        # clip the last episode
        buffer.clip_episode()

        # close logger
        logger.close()

    def _mutate_transition(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        variances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return observations, rewards

    def generate_replay_data_transition(self, dataset, max_save_num=1000, start_num=50, batch_size=16):
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
                assert self._impl._policy is not None
                transition_observations = np.stack([transition.observation for transition in transitions])
                transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                transition_actions = np.stack([transition.action for transition in transitions])
                transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
                transition_dists = self._impl._policy(transition_observations)
                transition_log_probs = torch.sum((transition_dists - transition_actions) ** 2, dim=1).to('cpu').detach().numpy().tolist()
                if self._experience_type == 'max_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1], reverse=True)]
                if self._experience_type == 'min_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1])]
            elif 'supervise' in self._experience_type:
                assert self._impl is not None
                assert self._impl._policy is not None
                i = 0
                transition_observations = np.stack([transition.observation for transition in transitions[i : i + batch_size]])
                transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                transition_dists = self._impl._policy(transition_observations)
                transition_qs = self._impl._q_func(transition_observations, transition_dists)
                transition_rewards = np.stack([transition.reward for transition in transitions[i : i + batch_size]])
                transition_rewards = torch.from_numpy(transition_rewards).to(self._impl.device)
                transition_next_observations = np.stack([transition.next_observation for transition in transitions[i : i + batch_size]])
                transition_next_observations = torch.from_numpy(transition_next_observations).to(self._impl.device)
                transition_next_dists = self._impl._policy(transition_observations)
                transition_next_qs = self._impl._q_func(transition_next_observations, transition_next_dists)
                transition_supervises = torch.abs(transition_rewards + self._gamma * (transition_next_qs - transition_qs).squeeze())
                if self._experience_type == 'max_supervise':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_supervises), key=lambda x: x[1], reverse=True)]
                if self._experience_type == 'min_supervise':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_supervises), key=lambda x: x[1])]
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
                        transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
                        mu_batch, logstd_batch = model.compute_stats(transition_observations, transition_actions)
                        mu_batches.append(mu_batch)
                        logstd_batches.append(logstd_batch)
                    mu_batch = sum(mu_batches) / len(mu_batches)
                    logstd_batch = sum(logstd_batches) / len(logstd_batches)
                    mu.append(mu_batch)
                    logstd.append(logstd_batch)
                    i += batch_size
                if i < transition_next_observations.shape[0]:
                    mu_batches = []
                    logstd_batches = []
                    for model in self._impl._dynamic._models:
                        transition_observations = np.stack([transition.observation for transition in transitions[i:]])
                        transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                        transition_actions = np.stack([transition.action for transition in transitions[i:]])
                        transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
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
                transition_log_probs = dists.log_prob(torch.cat([transition_next_observations, transition_rewards.unsqueeze(dim=1)], dim=1)).to('cpu').detach().numpy().tolist()
                if self._experience_type == 'max_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1], reverse=True)]
                if self._experience_type == 'min_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1])]
            else:
                raise NotImplementedError

            transitions = transitions[:max_save_num]
            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in transitions], dim=0).to(torch.float32).detach().to('cpu')
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in transitions], dim=0).to(torch.float32).detach().to('cpu')
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in transitions], dim=0).detach().to(torch.float32).to('cpu')
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in transitions], dim=0).detach().to(torch.float32).to('cpu')
            replay_episode_terminals = torch.ones_like(replay_terminals)
            replay_dataset = MDPDataset(replay_observations, replay_actions, replay_rewards, replay_terminals, replay_episode_terminals)
            return replay_dataset

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0

    def _get_rollout_horizon(self):
        return self._rollout_horizon

    def generate_replay_data_episode(self, dataset, max_save_num=1000, max_export_step=1000, start_num=1, batch_size=16, test=False):
        # max_save_num = all_max_save_num // 2
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
                    transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
                    transition_dists = self._impl._policy(transition_observations)
                    transition_log_probs = torch.sum((transition_dists - transition_actions) ** 2)
                    if self._experience_type[4:] == 'match_end':
                        episode_log_probs.append(transition_log_probs[-1])
                    elif self._experience_type[4:] == 'match_mean':
                        episode_log_probs.append(torch.mean(transition_log_probs))
                if self._experience_type[:3] == 'max':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_log_probs), key=lambda x: x[1], reverse=True)]
                elif self._experience_type[:3] == 'min':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_log_probs), key=lambda x: x[1])]
            elif self._experience_type[4:] in ['supervise_end', 'supervise_mean']:
                episode_supervise = []
                for episode in episodes:
                    transitions = episode.transitions
                    transition_observations = np.stack([transition.observation for transition in transitions])
                    transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                    transition_dists = self._impl._policy(transition_observations)
                    transition_qs = self._impl._q_func(transition_observations, transition_dists)
                    transition_rewards = np.stack([transition.reward for transition in transitions])
                    transition_rewards = torch.from_numpy(transition_rewards).to(self._impl.device)
                    transition_next_observations = np.stack([transition.next_observation for transition in transitions])
                    transition_next_observations = torch.from_numpy(transition_next_observations).to(self._impl.device)
                    transition_next_dists = self._impl._policy(transition_observations)
                    transition_next_qs = self._impl._q_func(transition_next_observations, transition_next_dists)
                    transition_supervise = torch.abs(transition_rewards + self._gamma * (transition_next_qs - transition_qs))
                    if self._experience_type[4:] == 'supervise_end':
                        episode_supervise.append(transition_supervise[-1])
                    elif self._experience_type[4:] == 'supervise_mean':
                        episode_supervise.append(torch.mean(transition_supervise))
                if self._experience_type[:3] == 'max':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_supervise), key=lambda x: x[1], reverse=True)]
                elif self._experience_type[:3] == 'min':
                    episodes = [i for i, _ in sorted(zip(episodes, episode_supervise), key=lambda x: x[1])]
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
                            transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
                            mu_batch, logstd_batch = model.compute_stats(transition_observations, transition_actions)
                            mu_batches.append(mu_batch)
                            logstd_batches.append(logstd_batch)
                        mu_batch = sum(mu_batches) / len(mu_batches)
                        logstd_batch = sum(logstd_batches) / len(logstd_batches)
                        mu.append(mu_batch)
                        logstd.append(logstd_batch)
                        i += batch_size
                    if i < transition_next_observations.shape[0]:
                        mu_batches = []
                        logstd_batches = []
                        for model in self._impl._dynamic._models:
                            transition_observations = np.stack([transition.observation for transition in transitions[i:]])
                            transition_observations = torch.from_numpy(transition_observations).to(self._impl.device)
                            transition_actions = np.stack([transition.action for transition in transitions[i:]])
                            transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)
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
            elif self._experience_type == 'max_q_mean':
                new_replay_diff_qs = []
                i = 0
                for episode in new_replay_dataset.episodes:
                    replay_observations = episode.observations.to(self._impl.device)
                    replay_actions = episode.actions.to(self._impl.device)
                    replay_qs = self._impl._q_func(replay_observations, replay_actions).detach().to(torch.float32).to('cpu')
                    replay_learned_qs = self._impl._q_func(replay_observations, self._impl._policy(replay_observations)).detach().to(torch.float32).to('cpu')
                    new_replay_diff_qs.append((replay_qs - replay_learned_qs).mean())
                new_replay_diff_qs = torch.stack(new_replay_diff_qs, dim=0)
            else:
                raise NotImplementedError

            all_transitions = [transition for episode in episodes for transition in episode.transitions]
            if max_save_num is not None:
                transitions = all_transitions[:max_save_num]
            else:
                transitions = all_transitions
            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in transitions], dim=0).to(torch.float32).to('cpu').numpy()
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in transitions], dim=0).to(torch.float32).to('cpu').numpy()
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in transitions], dim=0).to(torch.float32).to('cpu').numpy()
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in transitions], dim=0).to(torch.int64).to('cpu')
            replay_episode_terminals = torch.zeros_like(replay_terminals)
            i = -1
            for episode in episodes:
                i += episode.observations.shape[0]
                if i < replay_episode_terminals.shape[0]:
                    replay_episode_terminals[i] = 1
                else:
                    break
            replay_episode_terminals[-1] = 0
            replay_terminals = replay_terminals.numpy()
            replay_episode_terminals = replay_episode_terminals.numpy()
            replay_dataset = MDPDataset(replay_observations, replay_actions, replay_rewards, replay_terminals, replay_episode_terminals)
            return replay_dataset

    def generate_replay_data_online(self, env, test_id, test=False, max_save_num=1000):
        replay_observations = []
        replay_actions = []
        replay_rewards = []
        replay_next_observations = []
        replay_terminals = []
        replay_episode_terminals = []
        replay_policy_actions = []
        replay_qs = []
        try:
            env.reset_task(int(test_id))
        except:
            pass
        while len(replay_actions) < max_save_num:
            observation = env.reset()
            replay_observations.append(observation)
            observation = torch.from_numpy(observation).to(self._impl.device).unsqueeze(dim=0).to(torch.float32)
            episode_reward = 0.0

            i = 0
            while True:
                # take action
                action = self._impl._policy(observation)
                action = action.squeeze().cpu().detach().numpy()
                replay_actions.append(action)

                observation, reward, done, pos = env.step(action)
                replay_next_observations.append(observation)
                replay_rewards.append(reward)
                replay_terminals.append(done)

                if done:
                    replay_episode_terminals.append(0)
                    break
                if i > 1000:
                    replay_episode_terminals.append(1)
                    break
                replay_observations.append(observation)
                observation = torch.from_numpy(observation).to(self._impl.device).unsqueeze(dim=0).to(torch.float32)

                i += 1
                if i >= 2 and test:
                    break
        random_indexes = list(range(len(replay_actions)))
        random.shuffle(random_indexes)
        random_indexes = random_indexes[:max_save_num]
        replay_observations = torch.from_numpy(np.stack(replay_observations, axis=0)).to(self._impl.device).to(torch.float32)[random_indexes].numpy()
        replay_actions = torch.from_numpy(np.stack(replay_actions, axis=0)).to(self._impl.device).to(torch.float32)[random_indexes].numpy()
        replay_rewards = torch.from_numpy(np.array(replay_rewards)).to(torch.float32)[random_indexes].numpy()
        replay_terminals = torch.from_numpy(np.array(replay_terminals)).to(torch.int32)[random_indexes].numpy()
        replay_episode_terminals = torch.from_numpy(np.array(replay_episode_terminals)).to(torch.int32)[random_indexes].numpy()
        # online的情况下action就是由policy生成的。
        replay_dataset = MDPDataset(replay_observations, replay_actions, replay_rewards, replay_terminals, replay_episode_terminals)
        return replay_dataset

    def generate_replay(self, dataset_id, dataset, env, critic_replay_type, actor_replay_type, experience_type, max_save_num, max_export_step, test):

        start_time = time.perf_counter()
        if critic_replay_type in ['ewc', 'si', 'rwalk', 'none'] and actor_replay_type in ['ewc', 'si', 'rwalk', 'none'] or experience_type in ['none', 'all']:
            replay_dataset, save_dataset = None, None
        elif experience_type in ['random_transition', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'min_reward', 'min_match', 'min_supervise', 'min_model']:
            replay_dataset = self.generate_replay_data_transition(dataset, max_save_num=max_save_num)
            print(f"len(replay_dataset): {len(replay_dataset)}")
        elif experience_type in ['random_episode', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_model_end', 'min_model_mean', 'max_vq_diff', 'max_q']:
            replay_dataset = self.generate_replay_data_episode(dataset, max_save_num=max_save_num, max_export_step=max_export_step, test=test)
            print(f"len(replay_dataset): {len(replay_dataset)}")
            print(f"len(replay_dataset): {len(replay_dataset)}")
        elif experience_type == 'online':
            assert env is not None
            replay_dataset = self.generate_replay_data_online(env, dataset_id, test=test, max_save_num=max_save_num)
        else:
            print(f'experience_type: {experience_type}')
            raise NotImplementedError
        print(f'Select Replay Buffer Time: {time.perf_counter() - start_time}')
        return replay_dataset

    def select_replay(self, new_replay_dataset, old_replay_dataset, dataset_id, max_save_num, mix_type='random'):
        indices_old = None
        if mix_type == 'all':
            indices_new = np.arange(len(new_replay_dataset.episodes))
            if old_replay_dataset is not None:
                indices_old = np.arange(len(old_replay_dataset.episodes))
        elif mix_type == 'q_sample':
            new_replay_diff_qs = []
            temp_dataloader = DataLoader(TensorDataset(new_replay_dataset.obsrvations, new_replay_dataset.actions), batch_size=64, shuffle=False)
            for replay_observations_batch, replay_actions_batch in temp_dataloader:
                if self._impl_name in ['sacn', 'edac', 'iqln2']:
                    replay_q, _ = torch.min(self._impl._q_func(replay_observations_batch, replay_actions_batch), dim=0)
                else:
                    replay_q = self._impl._q_func(replay_observations_batch, replay_actions_batch)
                new_replay_diff_qs.append(replay_q)
            new_replay_diff_qs= torch.cat(replay_qs, dim=0)

            if old_replay_dataset is not None:
                new_replay_diff_qs = []
                temp_dataloader = DataLoader(TensorDataset(old_replay_dataset.observations, old_replay_dataset.actions), batch_size=64, shuffle=False)
                for replay_observations_batch, replay_actions_batch in temp_dataloader:
                    if self._impl_name in ['sacn', 'edac', 'iqln2']:
                        replay_q, _ = torch.min(self._impl._q_func(replay_observations_batch, replay_actions_batch), dim=0)
                    else:
                        replay_q = self._impl._q_func(replay_observations_batch, replay_actions_batch)
                    new_replay_diff_qs.append(replay_q)
                new_replay_diff_qs= torch.cat(replay_qs, dim=0)

                replay_diff_qs = torch.cat([new_replay_diff_qs, old_replay_diff_qs])
                replay_diff_qs = torch.clamp(replay_diff_qs, min = 1e-5)
                if max_save_num > replay_diff_qs.shape[0]:
                    indices = torch.arange(max_save_num)
                else:
                    indices = torch.multinomial(replay_diff_qs / torch.sum(replay_diff_qs), max_save_num)
                indices_new = indices[indices < len(new_replay_dataset)]
                indices_old = indices[indices >= len(new_replay_dataset)] - len(new_replay_dataset)
            else:
                replay_diff_qs = new_replay_diff_qs / torch.sum(new_replay_diff_qs)
                replay_diff_qs = torch.clamp(replay_diff_qs, min = 0)
                if max_save_num > replay_diff_qs.shape[0]:
                    indices_new = torch.arange(max_save_num)
                else:
                    indices_new = torch.multinomial(replay_diff_qs / torch.sum(replay_diff_qs), max_save_num)
                indices_old = None
        elif mix_type == 'q':
            new_replay_diff_qs = []
            i = 0
            for episode in new_replay_dataset.episodes:
                replay_observations = torch.from_numpy(episode.observations).to(self._impl.device)
                replay_actions = torch.from_numpy(episode.actions).to(self._impl.device)
                temp_dataloader = DataLoader(TensorDataset(replay_observations, replay_actions), batch_size=64, shuffle=False)
                replay_qs = []
                for replay_observations_batch, replay_actions_batch in temp_dataloader:
                    if self._impl_name in ['sacn', 'edac', 'iqln2']:
                        replay_q, _ = torch.min(self._impl._q_func(replay_observations_batch, replay_actions_batch), dim=0)
                    else:
                        replay_q = self._impl._q_func(replay_observations_batch, replay_actions_batch)
                    replay_qs.append(replay_q)
                replay_qs = torch.cat(replay_qs, dim=0)
                new_replay_diff_qs.append(replay_qs.mean())
            new_replay_diff_qs = torch.stack(new_replay_diff_qs, dim=0)

            if old_replay_dataset is not None:
                old_replay_diff_qs = []
                i = 0
                for episode in old_replay_dataset.episodes:
                    replay_observations = torch.from_numpy(episode.observations).to(self._impl.device)
                    replay_actions = torch.from_numpy(episode.actions).to(self._impl.device)
                    temp_dataloader = DataLoader(TensorDataset(replay_observations, replay_actions), batch_size=64, shuffle=False)
                    replay_qs = []
                    for replay_observations_batch, replay_actions_batch in temp_dataloader:
                        if self._impl_name in ['sacn', 'edac', 'iqln2']:
                            replay_q, _ = torch.min(self._impl._q_func(replay_observations_batch, replay_actions_batch), dim=0)
                        else:
                            replay_q = self._impl._q_func(replay_observations_batch, replay_actions_batch)
                        replay_qs.append(replay_q)
                    replay_qs = torch.cat(replay_qs, dim=0)
                    old_replay_diff_qs.append(replay_qs.mean())
                old_replay_diff_qs = torch.stack(old_replay_diff_qs, dim=0)

                replay_diff_qs = torch.cat([new_replay_diff_qs, old_replay_diff_qs])
                replay_diff_qs = torch.clamp(replay_diff_qs, min = 1e-5)
                if max_save_num > replay_diff_qs.shape[0]:
                    indices = torch.arange(max_save_num)
                else:
                    # indices = torch.multinomial(replay_diff_qs / torch.sum(replay_diff_qs), max_save_num)
                    _, indices = torch.topk(replay_diff_qs, max_save_num)
                indices_new = indices[indices < len(new_replay_dataset)]
                indices_old = indices[indices >= len(new_replay_dataset)] - len(new_replay_dataset)
            else:
                replay_diff_qs = new_replay_diff_qs / torch.sum(new_replay_diff_qs)
                replay_diff_qs = torch.clamp(replay_diff_qs, min = 0)
                if max_save_num > replay_diff_qs.shape[0]:
                    indices_new = torch.arange(max_save_num)
                else:
                    # indices_new = torch.multinomial(replay_diff_qs / torch.sum(replay_diff_qs), max_save_num)
                    _, indices_new = torch.topk(replay_diff_qs, max_save_num)
                indices_old = None
        elif mix_type == 'random':
            # slide_dataset_length = max_save_num // (dataset_id + 1)
            # indices_new = torch.randperm(slide_dataset_length).detach().cpu().numpy()
            # if old_replay_dataset is not None:
            #     old_slide_dataset_length = max_save_num // (dataset_id)
            #     indices_old = torch.cat([torch.arange(old_slide_dataset_length * i, old_slide_dataset_length * (i + 1), device=self._impl.device)[torch.randperm(slide_dataset_length)] for i in range(dataset_id)]).detach().cpu().numpy()
            indices_new = torch.randperm(len(new_replay_dataset.episodes))[: max_save_num // (dataset_id + 1)].detach().cpu().numpy()
            if old_replay_dataset is not None:
                indices_old = torch.randperm(len(old_replay_dataset.episodes))[max_save_num // (dataset_id + 1) :].detach().cpu().numpy()
            # slide_dataset_length = max_save_num // (dataset_id + 1)
            # indices_new = torch.randperm(slide_dataset_length).detach().cpu().numpy()
            # if old_replay_dataset is not None:
            #     old_slide_dataset_length = max_save_num // (dataset_id)
            #     indices_old = torch.cat([torch.arange(old_slide_dataset_length * i, old_slide_dataset_length * (i + 1), device=self._impl.device)[torch.randperm(slide_dataset_length)] for i in range(dataset_id)]).detach().cpu().numpy()
        else:
            raise NotImplementedError
        replay_dataset = self._generate_new_replay_dataset(new_replay_dataset, old_replay_dataset, indices_new, indices_old)
        return replay_dataset

    def _generate_new_replay_dataset(self, new_replay_dataset, old_replay_dataset, indices_new, indices_old):
        replay_terminals = []
        replay_episode_terminals = []
        new_episodes = new_replay_dataset.episodes
        if old_replay_dataset is not None:
            old_episodes = old_replay_dataset.episodes
            replay_observations = np.concatenate([new_episodes[i].observations for i in indices_new if i < len(new_episodes)] + [old_episodes[i].observations for i in indices_old if i < len(old_episodes)], axis=0)
            replay_actions = np.concatenate([new_episodes[i].actions for i in indices_new if i < len(new_episodes)] + [old_episodes[i].actions for i in indices_old if i < len(old_episodes)], axis=0)
            replay_rewards = np.concatenate([new_episodes[i].rewards for i in indices_new if i < len(new_episodes)] + [old_episodes[i].rewards for i in indices_old if i < len(old_episodes)], axis=0)
            for i in indices_new:
                terminals = np.zeros_like(replay_rewards, dtype=np.bool_)
                terminals[-1] = new_episodes[i].terminal
                episode_terminals = np.zeros_like(replay_rewards, dtype=np.bool_)
                episode_terminals[-1] = True
                replay_terminals.append(terminals)
                replay_episode_terminals.append(episode_terminals)
            for i in indices_old:
                terminals = np.zeros_like(replay_rewards, dtype=np.bool_)
                terminals[-1] = old_episodes[i].terminal
                episode_terminals = np.zeros_like(replay_rewards, dtype=np.bool_)
                episode_terminals[-1] = True
                replay_terminals.append(terminals)
                replay_episode_terminals.append(episode_terminals)
        else:
            replay_observations = np.concatenate([new_episodes[i].observations for i in indices_new if i < len(new_episodes)], axis=0)
            replay_actions = np.concatenate([new_episodes[i].actions for i in indices_new if i < len(new_episodes)], axis=0)
            replay_rewards = np.concatenate([new_episodes[i].rewards for i in indices_new if i < len(new_episodes)], axis=0)
            for i in indices_new:
                terminals = np.zeros_like(new_episodes[i].rewards, dtype=np.bool_)
                terminals[-1] = new_episodes[i].terminal
                episode_terminals = np.zeros_like(new_episodes[i].rewards, dtype=np.bool_)
                episode_terminals[-1] = True
                replay_terminals.append(terminals)
                replay_episode_terminals.append(episode_terminals)
        replay_terminals = np.concatenate(replay_terminals, axis=0)
        replay_episode_terminals = np.concatenate(replay_episode_terminals, axis=0)
        replay_dataset = MDPDataset(replay_observations, replay_actions, replay_rewards, replay_terminals, replay_episode_terminals)
        return replay_dataset

    def copy_load_model(
        self, pretrain_state_dict
    ) -> None:
        for key in _get_attributes(self._impl):
            if 'clone' in key:
                obj = getattr(self._impl, key)
                if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
                    remove_key = key[6:]
                    obj.load_state_dict(pretrain_state_dict[remove_key])
