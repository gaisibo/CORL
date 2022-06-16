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
from torch.utils.data import TensorDataset, DataLoader
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

from myd3rlpy.siamese_similar import similar_mb, similar_mb_euclid, similar_phi, similar_psi
# from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from myd3rlpy.algos.torch.co_impl import COImpl
from myd3rlpy.metrics.scorer import q_mean_scorer, q_replay_scorer
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class CO():
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
        task_id: str,
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
        eval_episodes_list: Optional[List[Dict[int, List[Episode]]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 1,
        eval_freq: int = int(5e3),
	scorers_list: Optional[List[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ]] = None,
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
                eval_episodes_list,
                save_interval,
                discount,
                start_timesteps,
                expl_noise,
                eval_freq,
                scorers_list,
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
        task_id: str,
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
        eval_episodes_list: Optional[List[Dict[int, List[Episode]]]] = None,
        save_interval: int = 1,
        discount: float = 0.99,
        start_timesteps : int = int(25e3),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
	scorers_list: Optional[List[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ]] = None,
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
            self._impl._impl_id = task_id
            LOG.debug("Models have been built.")
        else:
            self._impl.change_task(task_id)
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
        self.logger = logger

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

            if self._use_model:
                if n_dynamic_steps is not None:
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
                assert dynamic_state_dict is not None
                for key, value in dynamic_state_dict.items():
                    if 'model' in key or 'dynamic' in key:
                        try:
                            obj = getattr(self._impl, key)
                            if isinstance(obj, (torch.nn.Module)) or isinstance(obj, (torch.optim.Optimizer)):
                                obj = getattr(self._impl, key)
                                for name, input_param in dynamic_state_dict[key].items():
                                    try:
                                        param = obj.state_dict()[name]
                                    except:
                                        continue
                                    if len(input_param.shape) == 2:
                                        if input_param.shape[0] == param.shape[0] and input_param.shape[1] != param.shape[1]:
                                            input_param_append = torch.zeros(param.shape[0], param.shape[1] - input_param.shape[1]).to(param.dtype).to(param.device)
                                            torch.nn.init.kaiming_normal_(input_param_append)
                                            dynamic_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=1)
                                        if input_param.shape[0] != param.shape[0] and input_param.shape[1] == param.shape[1]:
                                            input_param_append = torch.zeros(param.shape[0] - input_param.shape[0], param.shape[1]).to(param.dtype).to(param.device)
                                            torch.nn.init.kaiming_normal_(input_param_append)
                                            dynamic_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=0)
                                    elif len(input_param.shape) == 1:
                                        if input_param.shape[0] != param.shape[0]:
                                            input_param_append = torch.zeros(param.shape[0] - input_param.shape[0]).to(param.dtype).to(param.device)
                                            dynamic_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=0)

                                obj.load_state_dict(dynamic_state_dict[key])
                                obj.requires_grad = True
                        except:
                            key = str(key)
                            obj = getattr(self._impl, key)
                            if isinstance(obj, (torch.nn.Module)):
                                obj = getattr(self._impl, key)
                                obj.load_state_dict(dynamic_state_dict[key])

            if task_id == '0' and pretrain_state_dict is not None:
                for key, value in pretrain_state_dict.items():
                    if 'actor' in key or 'critic' in key or 'policy' in key or 'q_func' in key:
                        try:
                            obj = getattr(self._impl, key)
                            if isinstance(obj, (torch.nn.Module)):
                                obj = getattr(self._impl, key)
                                for name, input_param in pretrain_state_dict[key].items():
                                    try:
                                        param = obj.state_dict()[name]
                                    except:
                                        continue
                                    if len(input_param.shape) == 2:
                                        if input_param.shape[0] == param.shape[0] and input_param.shape[1] != param.shape[1]:
                                            input_param_append = torch.zeros(param.shape[0], param.shape[1] - input_param.shape[1]).to(param.dtype).to(param.device)
                                            torch.nn.init.kaiming_normal_(input_param_append)
                                            pretrain_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=1)
                                        if input_param.shape[0] != param.shape[0] and input_param.shape[1] == param.shape[1]:
                                            input_param_append = torch.zeros(param.shape[0] - input_param.shape[0], param.shape[1]).to(param.dtype).to(param.device)
                                            torch.nn.init.kaiming_normal_(input_param_append)
                                            pretrain_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=0)
                                    elif len(input_param.shape) == 1:
                                        if input_param.shape[0] != param.shape[0]:
                                            input_param_append = torch.zeros(param.shape[0] - input_param.shape[0]).to(param.dtype).to(param.device)
                                            pretrain_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=0)

                                obj.load_state_dict(pretrain_state_dict[key])
                        except:
                            key = str(key)
                            obj = getattr(self._impl, key)
                            if isinstance(obj, (torch.nn.Module)):
                                obj = getattr(self._impl, key)
                                obj.load_state_dict(pretrain_state_dict[key])
            if pretrain_state_dict is not None:
                return

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
                if epoch > 1 and (test or (task_id == '0' and pretrain_state_dict is not None)):
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

                if scorers_list and eval_episodes_list:
                    for scorers, eval_episodes in zip(scorers_list, eval_episodes_list):
                        self._evaluate(eval_episodes, scorers, logger)

                # save metrics
                metrics = logger.commit(epoch, total_step)

                # save model parameters
                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                yield epoch, metrics

            # for EWC
            if self._replay_type in ['r_walk', 'ewc']:
                self._impl.ewc_r_walk_post_train_process(iterator)
            elif self._replay_type == 'si':
                self._impl.si_post_train_process
            elif self._replay_type == 'gem':
                self._impl.gem_post_train_process
            elif self._clone_actor and self._replay_type == 'bc':
                self._impl.bc_post_train_process

    def fit_dynamic(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        env: gym.envs = None,
        seed: int = None,
        n_dynamic_epochs: Optional[int] = None,
        n_dynamic_steps: Optional[int] = None,
        n_dynamic_steps_per_epoch: int = 10000,
        dynamic_state_dict: Optional[Dict[str, Any]] = None,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
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
            save_interval: interval to save parameters.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            list of result tuples (epoch, metrics) per epoch.
        """
        results = list(
            self.fitter_dynamic(
                task_id,
                dataset,
                env,
                seed,
                n_dynamic_epochs,
                n_dynamic_steps,
                n_dynamic_steps_per_epoch,
                dynamic_state_dict,
                save_metrics,
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
                # train_dynamics,
            )
        )
        return results

    def fitter_dynamic(
        self,
        task_id: int,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        env: gym.envs = None,
        seed: int = None,
        n_dynamic_epochs: Optional[int] = None,
        n_dynamic_steps: Optional[int] = 500000,
        n_dynamic_steps_per_epoch: int = 5000,
        dynamic_state_dict: Optional[Dict[str, Any]] = None,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        save_interval: int = 1,
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
            LOG.debug("Models have been built.")
        else:
            self._impl.change_task(task_id)
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

        if n_dynamic_steps is not None:
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
                print(key)
                if 'model' in key or 'dynamic' in key:
                    try:
                        obj = getattr(self._impl, key)
                        if isinstance(obj, (torch.nn.Module)) or isinstance(obj, (torch.optim.Optimizer)):
                            obj = getattr(self._impl, key)
                            for name, input_param in dynamic_state_dict[key].items():
                                try:
                                    param = obj.state_dict()[name]
                                except:
                                    continue
                                if len(input_param.shape) == 2:
                                    if input_param.shape[0] == param.shape[0] and input_param.shape[1] != param.shape[1]:
                                        input_param_append = torch.zeros(param.shape[0], param.shape[1] - input_param.shape[1]).to(param.dtype).to(param.device)
                                        torch.nn.init.kaiming_normal_(input_param_append)
                                        dynamic_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=1)
                                    if input_param.shape[0] != param.shape[0] and input_param.shape[1] == param.shape[1]:
                                        input_param_append = torch.zeros(param.shape[0] - input_param.shape[0], param.shape[1]).to(param.dtype).to(param.device)
                                        torch.nn.init.kaiming_normal_(input_param_append)
                                        dynamic_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=0)
                                elif len(input_param.shape) == 1:
                                    if input_param.shape[0] != param.shape[0]:
                                        input_param_append = torch.zeros(param.shape[0] - input_param.shape[0]).to(param.dtype).to(param.device)
                                        dynamic_state_dict[key][name] = torch.cat([input_param, input_param_append], dim=0)

                            obj.load_state_dict(dynamic_state_dict[key])
                    except:
                        key = str(key)
                        obj = getattr(self._impl, key)
                        if isinstance(obj, (torch.nn.Module)):
                            obj = getattr(self._impl, key)
                            obj.load_state_dict(dynamic_state_dict[key])
        total_step = 0
        for epoch in range(0, n_dynamic_epochs + 1):
            if epoch > 1 and test:
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
                if batch_num > 10 and test:
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

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger = None

    def _mutate_transition(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        variances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return observations, rewards

    def generate_new_data_replay(self, dataset, max_save_num=1000, max_export_time=100, max_export_step=1000, max_reward=None, real_action_size=1, real_observation_size=1, n_epochs=None, n_steps=500000,n_steps_per_epoch=5000, shuffle=True, save_metrics=True, experiment_name=None, with_timestamp=True, logdir='d3rlpy_logs', verbose=True, tensorboard_dir=None):
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._impl._policy
        assert self._impl._q_func

        start_observations = torch.from_numpy(np.stack([transitions[0].observation for episode in dataset.episodes for transitions in episode], axis=0)).to(self._impl.device)
        start_actions = self._impl._policy(start_observations)

        rets = []
        rets_num = 0
        # rollout
        # batch = TransitionMiniBatch(transitions)
        # observations = torch.from_numpy(batch.observations).to(self._impl.device)
        # actions = self._impl._policy(observations)
        # rewards = torch.from_numpy(batch.rewards).to(self._impl.device)
        generate_dataset = TensorDataset(start_observations, start_actions)
        generate_dataloader = DataLoader(generate_dataset, batch_size=self._batch_size, shuffle=True)
        for generate_batch_num, (observations, actions) in enumerate(generate_dataloader):
            for step in range(self._generate_step):

                # predict next state
                indexes = torch.randint(len(self._impl._dynamic._models), size=(observations.shape[0],))
                next_observations, next_rewards = self._impl._dynamic(observations[:, :real_observation_size], actions[:, :real_action_size], indexes)

                # sample policy action
                next_actions = self._impl._policy(next_observations)
                noise = 0.03 * (step + 1) / self._generate_step * torch.randn(next_actions.shape, device=self._impl.device)
                next_actions += noise

                rets.append({'observations': observations.to('cpu'), 'actions': actions.to('cpu'), 'rewards': next_rewards.to('cpu'), 'next_observations': next_observations.to('cpu'), 'terminals': torch.zeros(observations.shape[0], 1)})
                rets_num += observations.shape[0]

                observations = next_observations
                actions = next_actions
                rewards = next_rewards

        with torch.no_grad():
            replay_observations = torch.cat([transition['observations'] for transition in rets], dim=0)
            replay_actions = torch.cat([transition['actions'] for transition in rets], dim=0)
            replay_rewards = torch.cat([transition['rewards'] for transition in rets], dim=0)
            replay_next_observations = torch.cat([transition['next_observations'] for transition in rets], dim=0)
            replay_terminals = torch.cat([transition['terminals'] for transition in rets], dim=0)
            idx = torch.randperm(replay_observations.shape[0])
            replay_observations = replay_observations[idx].view(replay_observations.size())
            replay_actions = replay_actions[idx].view(replay_actions.size())
            replay_rewards = replay_rewards[idx].view(replay_rewards.size())
            replay_next_observations = replay_next_observations[idx].view(replay_next_observations.size())
            replay_terminals = replay_terminals[idx].view(replay_terminals.size())
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return replay_dataset, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device)).to('cpu')
                replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).to('cpu').detach()
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).to('cpu').detach()
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).to('cpu').detach()
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset

    def generate_replay_data_trajectory(self, dataset, episodes, start_index, max_save_num=1000, random_save_num=1000, max_export_time=1000, max_export_step=1000, real_action_size=1, real_observation_size=1, n_epochs=None, n_steps=500000,n_steps_per_epoch=5000, shuffle=True, with_generate='generate_model', test=False, indexes_euclid=None):
        assert self._impl is not None
        assert self._impl._policy is not None
        assert self._impl._q_func is not None
        assert self._impl._dynamic is not None
        if 'retrain' in self._sample_type:
            policy_state_dict = deepcopy(self._impl._policy.state_dict())
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

        observations = torch.from_numpy(np.stack([transition.observation for episode in episodes for transition in episode], axis=0))
        actions = torch.from_numpy(np.stack([transition.action for episode in episodes for transition in episode], axis=0))
        terminals = torch.from_numpy(np.stack([transition.terminal for episode in episodes for transition in episode], axis=0))
        terminals_stop = [len(episode.transitions) for episode in episodes]
        terminals_stop = [sum(terminals_stop[:i + 1]) - 1 for i in range(len(terminals_stop))]
        # 关键算法

        orl_lens = [0] + [len(episode) for episode in episodes[:start_index]]
        orl_indexes_all = []
        start = 0
        for orl_len in orl_lens:
            orl_indexes_all.append(start + orl_len)
            start += orl_len
        random.shuffle(orl_indexes_all)
        orl_ns_all = [0 for _ in orl_indexes_all]
        orl_batch_size = max_save_num // max_export_step
        orl_indexes_list = [orl_indexes_all[i:i+orl_batch_size] for i in range(0,len(orl_indexes_all) - 1,orl_batch_size)]
        orl_ns_list = [orl_ns_all[i:i+orl_batch_size] for i in range(0,len(orl_ns_all) - 1,orl_batch_size)]

        if indexes_euclid is None:
            near_next_observations = None

        export_time = 0
        # stop = False
        while len(orl_indexes_all) < max_save_num:
            # if test and stop:
            #     break
            for orl_indexes, orl_ns in zip(orl_indexes_list, orl_ns_list):
                if len(orl_indexes_all) >= max_save_num:
                    break
                # if test and stop:
                #     break
                start_indexes = orl_indexes
                start_observations = observations[np.array(orl_indexes)].to(self._impl.device)
                if self._sample_type == 'retrain_model':
                    start_actions = self._impl._policy(start_observations)
                print('next_while')
                export_step = 0
                epoch_loss = defaultdict(list)
                if 'retrain' in self._sample_type:
                    result_observations = []
                    result_actions = []
                    result_rewards = []
                    result_next_observations = []
                    result_terminals = []
                while export_step < max_export_step:
                    # if test and export_step >= 3:
                    #     stop = True
                    #     break
                    if len(orl_indexes) >= max_save_num:
                        break
                    print(f'next_step: {export_step}')
                    start_actions = self._impl._policy(start_observations)
                    if 'noise' in self._sample_type:
                        noise = 0.03 * max(1, (export_time / max_export_time)) * torch.randn(start_actions.shape, device=self._impl.device)
                        start_actions += noise

                    if indexes_euclid is not None:
                        start_indexes = np.array(start_indexes)
                        near_observations = observations[indexes_euclid[start_indexes]].to(self._impl.device)
                        near_actions = actions[indexes_euclid[start_indexes]].to(self._impl.device)
                        near_variances = []
                        for i in range(near_observations.shape[0]):
                            near_next_observations, _, variances = self._impl._dynamic.predict_with_variance(near_observations[i, :, :real_observation_size], near_actions[i, :, :real_action_size])
                            near_variances.append(torch.mean(variances))
                            mean_near_next_observations = torch.mean(near_next_observations, dim=1).unsqueeze(dim=1).expand(1, near_next_observations.shape[1], 1)
                            _, diff_mean_near_next_observations_indices = torch.max(torch.mean(near_next_observations - mean_near_next_observations, dim=2), dim=1)
                            near_next_observations = torch.stack([near_next_observations[i][diff_mean_near_next_observations_indices[i]] for i in range(diff_mean_near_next_observations_indices.shape[0])])
                        near_variances = torch.mean(torch.from_numpy(np.array(near_variances)).to(self._impl.device))
                        near_next_observations = torch.stack(near_next_observations)

                    if 'model' in with_generate:
                        # mus, logstds = [], []
                        # for model in self._impl._dynamic._models:
                        #     mu, logstd = model.compute_stats(start_observation, start_action)
                        #     mus.append(mu)
                        #     logstds.append(logstd)
                        # mus = torch.stack(mus, dim=1)
                        # mus += self._model_noise * torch.randn(mus.shape, device=self._impl.device)
                        # logstds = torch.stack(logstds, dim=1)
                        # noise = 0.03 * max(1, (export_time / max_export_time))
                        # logstds += noise

                        # 找到最接近中间的，也就是最准的。
                        start_next_observations, _, variances = self._impl._dynamic.predict_with_variance(start_observations[:, :real_observation_size], start_actions[:, :real_action_size])
                        variances = torch.mean(variances, dim=1)
                        mean_start_next_observations = torch.mean(start_next_observations, dim=1).unsqueeze(dim=1).expand(-1, start_next_observations.shape[1], -1)
                        _, diff_mean_start_next_observations_indices = torch.max(torch.mean(start_next_observations - mean_start_next_observations, dim=2), dim=1)
                        start_next_observations = torch.stack([start_next_observations[i][diff_mean_start_next_observations_indices[i]] for i in range(diff_mean_start_next_observations_indices.shape[0])])
                        if 'retrain' in self._sample_type:
                            start_next_actions = self._impl._policy(start_next_observations)

                            with torch.no_grad():
                                # original_reward
                                start_rewards = self._impl._q_func(start_observations, start_actions)
                                start_rewards -= self.gamma * self._impl._q_func(start_next_observations, start_next_actions)
                                orl_ns_tensor = torch.from_numpy(np.array(orl_ns))[:start_rewards.shape[0]].to(self._impl.device)
                                start_rewards -= self._orl_alpha * orl_ns_tensor.unsqueeze(dim=1)
                                start_rewards = start_rewards.detach()
                                # exploration reward

                            start_terminals = torch.zeros(start_observations.shape[0], 1).to(self._impl.device)
                            result_observations.append(start_observations)
                            result_actions.append(start_actions)
                            result_rewards.append(start_rewards)
                            result_next_observations.append(start_next_observations)
                            result_terminals.append(start_terminals)
                        # mus = mus[torch.arange(start_observations.shape[0]), torch.randint(len(self._impl._dynamic._models), size=(start_observations.shape[0],))]
                        # logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(self._impl._dynamic._models), size=(start_observations.shape[0],))]
                        # near_index = similar_mb(mus, logstds, near_observations, near_rewards.unsqueeze(dim=1), topk=1)
                        # 用dynamic的输出和dynamic的输出比较，减少一定的误差。
                        if indexes_euclid is not None:
                            near_indexes, _ = similar_mb_euclid(start_next_observations, near_next_observations, topk=1)
                            near_indexes.squeeze_()
                        else:
                            # 直接算的话会超内存，必须一个一个batch来。
                            batch_idx = 0
                            eval_batch_size = 10000
                            near_indexes = []
                            near_distances = []
                            near_variances = []
                            while batch_idx + eval_batch_size < observations.shape[0]:
                                near_observations = observations[batch_idx: batch_idx + eval_batch_size, :real_observation_size].to(self._impl.device)
                                near_actions = actions[batch_idx: batch_idx + eval_batch_size, :real_action_size].to(self._impl.device)
                                near_next_observations, _, variances_ = self._impl._dynamic.predict_with_variance(near_observations, near_actions)
                                near_variances.append(variances_)
                                mean_near_next_observations = torch.mean(near_next_observations, dim=1).unsqueeze(dim=1).expand(-1, near_next_observations.shape[1], -1)
                                _, diff_mean_near_next_observations_indices = torch.max(torch.mean(near_next_observations - mean_near_next_observations, dim=2), dim=1)
                                near_next_observations = torch.stack([near_next_observations[i][diff_mean_near_next_observations_indices[i]] for i in range(diff_mean_near_next_observations_indices.shape[0])])
                                near_indexes_, near_distances_ = similar_mb_euclid(start_next_observations, near_next_observations, topk=1)
                                near_indexes.append(near_indexes_)
                                near_distances.append(near_distances_)
                                batch_idx += eval_batch_size
                            near_indexes = torch.cat(near_indexes, dim=1)
                            near_distances = torch.cat(near_distances, dim=1)
                            near_distances, near_indexes_inner = torch.topk(near_distances, 1, largest=False, dim=1)
                            near_indexes = near_indexes.gather(1, near_indexes_inner).squeeze()
                            near_variances = torch.mean(torch.cat(near_variances, dim=0))
                        # 仅在dynamic model足够准确的情况下跳转，否则不动。
                        start_next_indexes = torch.from_numpy(np.array(start_indexes).astype(np.int64)).to(self._impl.device) + 1
                        near_indexes = torch.where(variances < near_variances, start_next_indexes, near_indexes)
                        # near_indexes = [near_index + 1 for near_index in near_indexes]
                    # elif 'siamese' in with_generate:
                    #     near_indexes, _, _ = similar_phi(start_observation, start_action[:, :real_action_size], near_observations, near_actions, self._impl._phi, topk=1)
                    else:
                        raise NotImplementedError
                    near_indexes = near_indexes.detach().cpu().numpy()
                    if len(near_indexes) == 0:
                        break
                    start_indexes = []
                    for start_index in near_indexes:
                        if start_index not in terminals_stop:
                            start_indexes.append(start_index)
                        else:
                            print(f'start_indexes {start_index} finish')
                    start_indexes = list(set(start_indexes))
                    for start_index in start_indexes:
                        if start_index in orl_indexes_all:
                            new_index = orl_indexes_all.index(start_index)
                            orl_ns_all[new_index] += 1
                        else:
                            orl_indexes_all.append(start_index)
                            orl_ns_all.append(1)
                    print(f'start_indexes: {len(start_indexes)}')
                    if len(start_indexes) == 0:
                        break
                    start_observations = observations[start_indexes].to(self._impl.device)
                    if len(start_observations.shape) == 1:
                        start_observations.unsqueeze(dim=0)
                    export_step += 1
                export_time += 1
            if 'retrain' in self._sample_type:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator.reset()
                    batch = next(iterator)
                batch_new = dict()
                if self._sample_type == 'retrain_actor':
                    batch_new['observations'] = torch.cat([torch.from_numpy(batch.observations).to(self._impl.device)] + result_observations, dim=0).detach()
                    batch_new['actions'] = torch.cat([torch.from_numpy(batch.actions[:, :real_action_size]).to(self._impl.device)] + result_actions, dim=0).detach()
                    batch_new['rewards'] = torch.cat([torch.from_numpy(batch.rewards).to(self._impl.device)] + result_rewards, dim=0).detach()
                    batch_new['next_observations'] = torch.cat([torch.from_numpy(batch.next_observations).to(self._impl.device)] + result_next_observations, dim=0).detach()
                    batch_new['terminals'] = torch.cat([torch.from_numpy(batch.terminals).to(self._impl.device)] + result_terminals, dim=0).detach()
                elif self._sample_type == 'retrain_model':
                    batch_new['observations'] = torch.from_numpy(batch.observations).to(self._impl.device).detach()
                    batch_new['actions'] = torch.from_numpy(batch.actions[:, :real_action_size]).to(self._impl.device)
                    batch_new['rewards'] = torch.from_numpy(batch.rewards).to(self._impl.device)
                    batch_new['next_observations'] = torch.from_numpy(batch.next_observations).to(self._impl.device)
                    batch_new['terminals'] = torch.from_numpy(batch.terminals).to(self._impl.device)
                batch_new = Struct(**batch_new)
                batch_new.n_steps = 1
                batch_new.masks = None
                batch_new.device = self._impl.device
                batch_new = cast(TorchMiniBatch, batch_new)
                if self._sample_type == 'retrain_actor':
                    loss = self._retrain_actor_update(batch_new)
                elif self._sample_type == 'retrain_model':
                    batch_retrain['observations'] = torch.cat(result_observations, dim=0).detach()
                    batch_retrain['actions'] = torch.cat(result_actions, dim=0).detach()
                    batch_retrain['rewards'] = torch.cat(result_rewards, dim=0).detach()
                    batch_retrain['next_observations'] = torch.cat(result_next_observations, dim=0).detach()
                    batch_retrain['terminals'] = torch.cat(result_terminals, dim=0).detach()
                    batch_retrain = Struct(**batch_retrain)
                    batch_retrain.n_steps = 1
                    batch_retrain.masks = None
                    batch_retrain.device = self._impl.device
                    batch_retrain = cast(TorchMiniBatch, batch_retrain)
                    loss = self._retrain_model_update(batch_new, batch_retrain)
        if self._sample_type == 'retrain_actor':
            self._impl._policy.load_state_dict(policy_state_dict)

        all_transitions = np.array([transition for episode in episodes for transition in episode])
        random.shuffle(orl_indexes_all)
        if len(orl_indexes_all) >= max_save_num:
            orl_indexes_all = orl_indexes_all[:max_save_num]
        orl_indexes_random = list(set(range(len(all_transitions))) - set(orl_indexes_all))
        random.shuffle(orl_indexes_random)
        orl_indexes_random = orl_indexes_random[:random_save_num]
        test_transitions = [all_transitions[orl_index] for orl_index in orl_indexes_all]
        orl_indexes_all += orl_indexes_random
        transitions = [all_transitions[orl_index] for orl_index in orl_indexes_all]
        if with_generate == 'generate_model':
            return self.generate_new_data_replay(transitions, max_save_num=max_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size)
        else:
            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in transitions], dim=0).to('cpu')
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in transitions], dim=0).to('cpu')
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in transitions], dim=0).to('cpu')
            replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in transitions], dim=0).to('cpu')
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in transitions], dim=0).to('cpu')
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return transitions, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device)).to('cpu')
                # replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                replay_qs = torch.zeros(replay_observations.shape[0])
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach().to('cpu')
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset

    def generate_replay_data_transition(self, dataset, max_save_num=1000, start_num=50, real_observation_size=1, real_action_size=1, batch_size=16, with_generate='none', indexes_euclid=None, distances_euclid=None, d_threshold=None):
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
                transition_actions = torch.from_numpy(transition_actions).to(self._impl.device)[:, :real_action_size]
                transition_dists = self._impl._policy(transition_observations)
                transition_log_probs = torch.sum((transition_dists - transition_actions) ** 2, dim=1).to('cpu').detach().numpy().tolist()
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
                transition_log_probs = dists.log_prob(torch.cat([transition_next_observations, transition_rewards.unsqueeze(dim=1)], dim=1)).to('cpu').detach().numpy().tolist()
                if self._experience_type == 'max_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1], reverse=True)]
                if self._experience_type == 'min_match':
                    transitions = [i for i, _ in sorted(zip(transitions, transition_log_probs), key=lambda x: x[1])]
            elif self._experience_type == 'coverage':
                assert indexes_euclid is not None and distances_euclid is not None
                distances_quantile = torch.quantile(distances_euclid, q=torch.arange(0, 1.01, 0.1), dim=0)
                print(f"distances_quantile: {distances_quantile}")
                assert False
                near_n = torch.sum(torch.where(distances_euclids < d_threshold, torch.ones_like(distances_euclids), torch.zeros_like(distances_euclids)), dim=0)
                transitions = [i for i, _ in sorted(zip(transitions, near_n), key=lambda x: x[1])]
            else:
                raise NotImplementedError
            if with_generate == 'generate':
                transitions = transitions[:max_save_num // self._generate_step]
                return self.generate_new_data_replay(transitions, max_save_num=max_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size)

            transitions = transitions[:max_save_num]
            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in transitions], dim=0).to('cpu')
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in transitions], dim=0).to('cpu')
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in transitions], dim=0).to('cpu')
            replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in transitions], dim=0).to('cpu')
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in transitions], dim=0).to('cpu')
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return transitions, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device)).to('cpu')
                replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach().to('cpu')
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset

    def _is_generating_new_data(self) -> bool:
        return self._grad_step % self._rollout_interval == 0

    def _get_rollout_horizon(self):
        return self._rollout_horizon

    def generate_replay_data_episode(self, dataset, all_max_save_num=1000, start_num=1, real_observation_size=1, real_action_size=1, batch_size=16, with_generate='none', test=False, indexes_euclid=None):
        # max_save_num = all_max_save_num // 2
        # random_save_num = all_max_save_num - max_save_num
        max_save_num = all_max_save_num
        random_save_num = 0
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
                    elif self._experience_type[4:] == 'match_mean':
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
                        i += batch_size
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
            if with_generate == 'generate':
                transitions = [transition for episode in episodes for transition in episode.transitions]
                transitions = transitions[:max_save_num // self._generate_step]
                return self.generate_new_data_replay(transitions, max_save_num=max_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size)
            if with_generate in ['generate_model', 'model']:
                assert indexes_euclid is None
                select_num = 0
                if with_generate == 'generate_model':
                    given_length = max_save_num // self._generate_step * self._select_time
                else:
                    given_length = max_save_num * self._select_time
                saved = False
                episode_num = 0
                start_index = 0
                for episode_num, episode in enumerate(episodes):
                    select_num += len(episode.transitions)
                    if select_num >= max_save_num:
                        if not saved:
                            start_index = episode_num
                            saved = True
                    if select_num >= given_length:
                        break
                if not saved:
                    start_index = episode_num
                episodes = episodes[:episode_num]
                return self.generate_replay_data_trajectory(dataset, episodes, start_index, max_save_num=max_save_num, random_save_num=random_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size, with_generate=with_generate, test=test, indexes_euclid=indexes_euclid)

            all_transitions = [transition for episode in episodes for transition in episode.transitions]
            transitions = all_transitions[:max_save_num]
            random_transitions = all_transitions[max_save_num:]
            random.shuffle(random_transitions)
            transitions += random_transitions[:random_save_num]
            replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in transitions], dim=0).to('cpu')
            replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in transitions], dim=0).to('cpu')
            replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in transitions], dim=0).to('cpu')
            replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in transitions], dim=0).to('cpu')
            replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in transitions], dim=0).to('cpu')
            if self._replay_type != 'bc':
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
                return transitions, replay_dataset
            else:
                replay_policy_actions = self._impl._policy(replay_observations.to(self._impl.device)).to('cpu')
                # replay_qs = self._impl._q_func(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                replay_qs = torch.zeros(replay_observations.shape[0])
                if self._use_phi:
                    replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                    replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach().to('cpu')
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
                else:
                    replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
                return replay_dataset, replay_dataset

    def generate_replay_data_online(self, env, max_save_num=1000, real_observation_size=1, real_action_size=1):
        replay_observations = []
        replay_actions = []
        replay_rewards = []
        replay_next_observations = []
        replay_terminals = []
        replay_policy_actions = []
        replay_qs = []
        while len(replay_actions) < max_save_num:
            observation = env.reset()
            observation = torch.from_numpy(observation).to(self._impl.device).unsqueeze(dim=1)
            replay_observations.append(observation)
            print(f'observation: {observation.shape}')
            episode_reward = 0.0

            i = 0
            while True:
                # take action
                action = self._impl._policy(observation)
                replay_actions.append(actions)
                action = action.cpu().detach().numpy()

                observation, reward, done, pos = env.step(action)
                observation = torch.from_numpy(observation).to(self._impl.device).unsqueeze(dim=1)
                replay_next_observations.append(observation)
                reward = torch.from_numpy(reward).to(self._impl.device).unsqueeze(dim=1)
                replay_rewards.append(reward)
                terminals = torch.from_numpy(done).to(self._impl.device).unsqueeze(dim=1)
                replay_terminals.append(terminals)

                if done:
                    break
                if i > 1000:
                    break
                replay_observations.append(observation)

                i += 1
        if self._replay_type == 'bc':
            transitions = [Transition(real_observation_size, real_action_size, observation, action, reward, next_observation, terminal) for observation, action, reward, next_observation, terminal in zip(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)]
        random_indexes = list(range(len(replay_actions)))
        random.shuffle(random_indexes)
        random_indexes = random_indexes[:max_save_num]
        replay_observations = torch.cat(replay_observations, dim=0)[random_indexes]
        replay_next_observations = torch.cat(replay_next_observations, dim=0)[random_indexes]
        replay_actions = torch.cat(replay_actions, dim=0)[random_indexes]
        replay_terminals = torch.cat(replay_terminals, dim=0)[random_indexes].detach().to('cpu')
        replay_rewards = torch.cat(replay_rewards, dim=0)[random_indexes].detach().to('cpu')
        replay_observations = replay_observations.detach().to('cpu')
        replay_actions = replay_actions.detach().to('cpu')
        if self._replay_type != 'bc':
            transitions = [transitions[random_index] for random_index in random_indexes]
            replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals)
            return transitions, replay_dataset
        else:
            # online的情况下action就是由policy生成的。
            replay_policy_actions = replay_actions
            replay_qs = self._impl._q_func(replay_observations, replay_actions).detach().to('cpu')
            if self._use_phi:
                replay_phis = self._impl._phi(replay_observations.to(self._impl.device), replay_actions[:, :real_action_size].to(self._impl.device)).detach().to('cpu')
                replay_psis = self._impl._psi(replay_observations.to(self._impl.device)).detach().to('cpu')
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
            else:
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
            return replay_dataset, replay_dataset

    def generate_replay(self, task_id, origin_datasets, envs, replay_type, experience_type, replay_datasets, save_datasets, max_save_num, real_action_size, real_observation_size, generate_type, indexes_euclid, distances_euclid, d_threshold, with_generate, test, model_path, algos_name, scorers_list, eval_episodes_list, learned_tasks):

        if int(task_id) != len(origin_datasets.keys()) - 1:
            start_time = time.perf_counter()
            if replay_type in ['ewc', 'si', 'r_walk']:
                replay_datasets[task_id], save_datasets[task_id] = None, None
            elif experience_type in ['random_transition', 'max_reward', 'max_match', 'max_model', 'min_reward', 'min_match', 'min_model']:
                replay_datasets[task_id], save_datasets[task_id] = self.generate_replay_data_transition(origin_datasets[task_id], max_save_num=max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size, with_generate=generate_type, indexes_euclid=indexes_euclid, distances_euclid=distances_euclid, d_threshold=d_threshold)
                print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
            elif experience_type in ['random_episode', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_model_end', 'max_model_mean', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_model_end', 'min_model_mean']:
                replay_datasets[task_id], save_datasets[task_id] = self.generate_replay_data_episode(origin_datasets[task_id], all_max_save_num=max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size, with_generate=generate_type, test=test, indexes_euclid=indexes_euclid)
                print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
            elif experience_type == 'generate':
                replay_datasets[task_id], save_datasets[task_id] = self.generate_new_data_replay(origin_datasets[task_id], max_save_num=max_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size)
                print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
            elif experience_type == 'model':
                episodes = origin_datasets[task_id].episodes
                episode_num = 0
                saved = False
                select_num = 0
                start_index = 0
                for episode_num, episode in enumerate(episodes):
                    select_num += len(episode.transitions)
                    if select_num >= max_save_num:
                        if not saved:
                            start_index = episode_num
                            saved = True
                if not saved:
                    start_index = episode_num
                replay_datasets[task_id], save_datasets[task_id] = self.generate_replay_data_trajectory(origin_datasets[task_id], episodes, start_index, max_save_num=max_save_num, random_save_num=0, real_observation_size=real_observation_size, real_action_size=real_action_size, with_generate=generate_type, test=test, indexes_euclid=indexes_euclid)
                print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
            elif experience_type == 'online':
                assert envs is not None
                replay_datasets[task_id], save_datasets[task_id] = self.generate_replay_data_online(envs[task_id], max_save_num=max_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size)
            else:
                replay_datasets[task_id], save_datasets[task_id] = None, None
            print(f'Select Replay Buffer Time: {time.perf_counter() - start_time}')
            if save_datasets[task_id] is not None:
                torch.save(save_datasets[task_id], f=model_path + algos_name + '_' + str(task_id) + '_datasets.pt')
        mean_scorers = scorers_list[1]
        mean_eval_episodes = eval_episodes_list[1]
        mean_qs = []
        for mean_scorer, mean_eval_episode in zip(mean_scorers, mean_eval_episodes):
            mean_qs.append(mean_scorer(mean_eval_episode))
        replay_scorers = [q_replay_scorer(real_action_size=real_action_size, test_id=str(n)) for n in learned_tasks[:-1]]
        replay_qs = []
        for replay_scorer, replay_eval_episode in zip(replay_scorers, save_datasets):
            replay_qs.append(replay_scorer(replay_eval_episode))
        no_replay_qs = [(mean_q * origin_dataset.observations.shape[0] - max_save_num * replay_q) / (origin_dataset.observations.shape[0] - max_save_num) for mean_q, origin_dataset, replay_q in zip(mean_qs, origin_datasets, replay_qs)]
        for i, no_replay_q in enumerate(no_replay_qs):
            self.logger.add_metric(f'no_replay_q{i}', no_replay_q)
