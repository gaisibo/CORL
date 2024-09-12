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

from d3rlpy.logger import LOG
from d3rlpy.argument_utility import (
    EncoderArg,
)
from d3rlpy.torch_utility import TorchMiniBatch, _get_attributes
from d3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch, Transition
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.combo import COMBO
from d3rlpy.constants import (
    IMPL_NOT_INITIALIZED_ERROR,
)
from d3rlpy.base import LearnableBase
from d3rlpy.iterators import TransitionIterator
from d3rlpy.iterators.random_iterator import RandomIterator
from d3rlpy.iterators.round_iterator import RoundIterator
import gym

from myd3rlpy.algos.torch.base_model_impl import BaseModelImpl
# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class BaseModel():
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

    _impl: BaseModelImpl
    # actor必须被重放，没用选择。
    _tau: float
    _use_gpu: Optional[Device]
    _generate_step: int
    _select_time: int
    _model_noise: float

    def __init__(
        self,
        *,
        model_learning_rate: float = 1e-4,
        model_optim_factory: OptimizerFactory = AdamFactory(),
        model_encoder_factory: EncoderArg = "default",
        impl_name = 'co',
        model_n_ensembles = 5,
        model_noise = 0,
        retrain_model_alpha = 1,
        **kwargs: Any
    ):

        self._impl_name = impl_name
        # self._n_train_dynamics = n_train_dynamics

        self._begin_grad_step = 0

        self._model_learning_rate = model_learning_rate
        self._model_optim_factory = model_optim_factory
        self._model_encoder_factory = model_encoder_factory
        self._model_n_ensembles = model_n_ensembles
        self._retrain_model_alpha = retrain_model_alpha
        self._model_noise = model_noise

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int, task_id: int
    ) -> None:
        assert self._impl_name in ['co', 'gemco', 'agemco']
        self._impl = BaseModelImpl(
            model_learning_rate=self._model_learning_rate,
            model_optim_factory=self._model_optim_factory,
            model_encoder_factory=self._model_encoder_factory,
            use_gpu=self._use_gpu,
            model_n_ensembles=self._model_n_ensembles,
            retrain_model_alpha=self._retrain_model_alpha,
        )
        self._impl.build(task_id)

    def update(self, batch: TransitionMiniBatch) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._update(batch)
        self._grad_step += 1
        return loss

    # 注意欧氏距离最近邻被塞到actions后面了。
    def _update(self, batch: TransitionMiniBatch) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        model_loss = self._impl.update_model(batch)
        metrics.update({"model_loss": model_loss})

        return metrics

    def fit(
        self,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        n_dynamic_epochs: Optional[int] = None,
        n_dynamic_steps: Optional[int] = None,
        n_dynamic_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = False,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        save_interval: int = 1,
        shuffle: bool = True,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        real_action_size: int = 0,
        real_observation_size: int = 0,
        test: bool = False,
        epoch_num: Optional[int] = None,
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
        assert dataset is not None
        results = list(
            self.fitter(
                dataset,
                n_dynamic_epochs,
                n_dynamic_steps,
                n_dynamic_steps_per_epoch,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                save_interval,
                shuffle,
                callback,
                real_action_size,
                real_observation_size,
                test,
                epoch_num,
                # train_dynamics,
            )
        )
        return results

    def fitter(
        self,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        n_dynamic_epochs: Optional[int] = None,
        n_dynamic_steps: Optional[int] = 500000,
        n_dynamic_steps_per_epoch: int = 5000,
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
        if n_dynamic_steps is not None:
            assert n_dynamic_steps >= n_dynamic_steps_per_epoch
            n_epochs = n_dynamic_steps // n_dynamic_steps_per_epoch
            iterator = RandomIterator(
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

            for batch_num, itr in enumerate(range_gen):
                if batch_num > 10 and test:
                    break
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)
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