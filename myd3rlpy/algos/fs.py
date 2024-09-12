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
from numpy.matrixlib.defmatrix import N
from tqdm.auto import tqdm
from tqdm.auto import trange
import numpy as np
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.distributions.normal import Normal
from d3rlpy.online.buffers import Buffer
from d3rlpy.metrics.scorer import evaluate_on_environment

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
from d3rlpy.torch_utility import TorchMiniBatch, _get_attributes, hard_sync, map_location, get_state_dict
from d3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch, Transition
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase, AlgoImplBase
from d3rlpy.algos.torch.base import TorchImplBase
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

# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from myd3rlpy.algos.torch.state_vae_impl import StateVAEImpl
from myd3rlpy.metrics.scorer import q_mean_scorer, q_replay_scorer
from myd3rlpy.models.vaes import create_vae_factory
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class FSBase():
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

    def update(self, batch_random: TransitionMiniBatch, batch_outer: TransitionMiniBatch, batch_inner: TransitionMiniBatch, online: bool = False, batch_num: int=0, total_step: int=0, coldstart_steps: Optional[int] = None, score=False) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        feature, loss = self._update(batch_random, batch_outer, batch_inner, online, batch_num, total_step, coldstart_steps, score=False)
        self._grad_step += 1
        return feature, loss

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

    def make_iterator(self, dataset, n_steps, n_steps_per_epoch, n_epochs, shuffle):
        iterator: TransitionIterator
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

        if n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = RandomIterator(
                transitions,
                n_steps_per_epoch,
                batch_size=self._batch_size * 2,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                transitions,
                batch_size=self._batch_size * 2,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")

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
        return iterator, n_epochs

    def fit(
        self,
        dataset_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        iterator: Optional[TransitionIterator] = None,
        n_epochs: Optional[int] = None,
        coldstart_steps: Optional[int] = None,
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
        score=False,
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
                n_epochs,
                coldstart_steps,
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
                score,
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
        n_epochs: Optional[int] = None,
        coldstart_steps: Optional[int] = None,
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
        score=False,
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
            if epoch > 2 and test:
                break

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator) // 2),
                disable=not show_progress,
                desc=f"Epoch {epoch}/{n_epochs}",
            )

            iterator.reset()
            features = []
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
                        batch_outer = next(iterator)
                        batch_inner = next(iterator)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        feature, loss = self.update(batch_inner, batch_outer, batch_inner, batch_num=batch_num, total_step=total_step, coldstart_steps=coldstart_steps, score=score)
                        features.append(feature)
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
                    self._evaluate(eval_episodes, rename_scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics, features

    def after_learn(self, iterator, experiment_name, scorers_list, eval_episodes_list, logdir='d3rlpy_logs'):
        # for EWC
        if self._critic_replay_type in ['rwalk', 'ewc']:
            self._impl.critic_ewc_rwalk_post_train_process(iterator)
        # elif self._critic_replay_type == 'si':
        #     self._impl.critic_si_post_train_process()
        # elif self._critic_replay_type == 'gem':
        #     self._impl.gem_post_train_process()
        if self._actor_replay_type in ['rwalk', 'ewc']:
            self._impl.actor_ewc_rwalk_post_train_process(iterator)
        # if self._embed_replay_type in ['rwalk', 'ewc']:
        #     self._embed._ewc_rwalk_post_train_process(iterator)
        # elif self._actor_replay_type == 'si':
        #     self._impl.actor_si_post_train_process()
        if self._impl_name in ['mgcql', 'mqcql', 'mrcql']:
            self._impl.match_prop_post_train_process(iterator)
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

    def online_fit(
        self,
        env: gym.envs,
        eval_env: gym.envs,
        buffer: Buffer,
        n_steps: int = 10000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
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
        observation = env.reset()
        rollout_return = 0.0


        for total_step in xrange(1, n_steps + 1):
            if total_step > 1000 and test:
                break
            with logger.measure_time("step"):
                observation = observation.astype("f4")
                fed_observation = observation

                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_steps:
                        action = env.action_space.sample()
                    else:
                        action = self.sample_action([fed_observation])[0]

                # step environment
                with logger.measure_time("environment_step"):
                    next_observation, reward, terminal, info = env.step(action)
                    rollout_return += reward

                # special case for TimeLimit wrapper
                if timelimit_aware and "TimeLimit.truncated" in info:
                    clip_episode = True
                    terminal = False
                else:
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
                    observation = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    rollout_return = 0.0
                    # for image observation
                else:
                    observation = next_observation

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch

                if total_step > update_start_step and len(buffer) > self._batch_size * 2:
                    if total_step % update_interval == 0:
                        # sample mini-batch
                        with logger.measure_time("sample_batch"):
                            batch = buffer.sample(
                                batch_size=self._batch_size * 2,
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

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_scorer:
                    logger.add_metric("evaluation", eval_scorer(self))

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

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0

    def _get_rollout_horizon(self):
        return self._rollout_horizon

    def copy_load_model(
        self, pretrain_state_dict
    ) -> None:
        for key in _get_attributes(self._impl):
            if 'clone' in key:
                obj = getattr(self._impl, key)
                if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
                    remove_key = key[6:]
                    obj.load_state_dict(pretrain_state_dict[remove_key])
