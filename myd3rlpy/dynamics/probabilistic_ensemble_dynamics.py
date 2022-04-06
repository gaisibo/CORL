from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import gym
import numpy as np
import torch
from tqdm.auto import tqdm

from d3rlpy.argument_utility import (
    ActionScalerArg,
    ScalerArg,
    UseGPUArg,
    EncoderArg,
)
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    ActionSpace,
)
from d3rlpy.context import disable_parallel
from d3rlpy.dataset import Episode, MDPDataset, Transition
from d3rlpy.iterators import RandomIterator, RoundIterator, TransitionIterator
from d3rlpy.logger import LOG
from d3rlpy.base import LearnableBase
from d3rlpy.argument_utility import ActionScalerArg, ScalerArg
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from d3rlpy.dynamics.torch.probabilistic_ensemble_dynamics_impl import ProbabilisticEnsembleDynamicsImpl

from myd3rlpy.siamese_similar import similar_mb


class ProbabilisticEnsembleDynamics(ProbabilisticEnsembleDynamics):

    """ProbabilisticEnsembleDynamics with following sample"""

    def __init__(
        self,
        *,
        task_id: int,
        original: torch.Tensor,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(weight_decay=1e-4),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        n_ensembles: int = 5,
        variance_type: str = "max",
        discrete_action: bool = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        use_gpu: UseGPUArg = False,
        impl: Optional[ProbabilisticEnsembleDynamicsImpl] = None,
        topk: int = 4,
        network = None,
        id_size: int = 7,
        **kwargs: Any
    ):
        super().__init__(
            learning_rate = learning_rate,
            optim_factory = optim_factory,
            encoder_factory = encoder_factory,
            batch_size = batch_size,
            n_frames = n_frames,
            n_ensembles = n_ensembles,
            variance_type = variance_type,
            discrete_action = discrete_action,
            scaler = scaler,
            action_scaler = action_scaler,
            use_gpu = use_gpu,
            impl = impl,
            topk = topk,
            kwargs = kwargs,
        )
        self._original = original
        self._topk = topk
        self._network = network
        self._task_id = task_id
        self._id_size = id_size

    def fit(
        self,
        dataset: Union[List[Episode], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        pretrain: bool = True,
        test: bool = True,
    ) -> List[Tuple[int, Dict[str, float]]]:
        results = list(
            self.fitter(
                dataset,
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
                pretrain,
                test,
            )
        )
        return results

    def fitter(
        self,
        dataset: Union[List[Episode], MDPDataset],
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
        pretrain: bool = True,
        test: bool = True,
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

        iterator: TransitionIterator
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

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            transition = iterator.transitions[0]
            action_size = transition.get_action_size()
            observation_shape = tuple(transition.get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )
            assert self._impl.device
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):

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

    # def generate_new_data_trajectory(self, dataset, max_export_time = 0, max_reward=None):
    #     if self._network is not None:
    #         # 关键算法
    #         _original = torch.from_numpy(self._original).to(self._impl.device)
    #         task_id_numpy = np.eye(self._id_size)[self._task_id].squeeze()
    #         task_id_numpy = torch.from_numpy(np.broadcast_to(task_id_numpy, (_original.shape[0], self._id_size))).to(torch.float32).to(self._impl.device)
    #         original_observation = torch.cat([_original, task_id_numpy], dim=1)
    #         with torch.no_grad():
    #             original_action = self._network._impl._policy(original_observation)
    #         replay_indexes = []
    #         new_transitions = []

    #         export_time = 0
    #         start_indexes = torch.zeros(0)
    #         while start_indexes.shape[0] != 0 and original_observation is not None and export_time < max_export_time:
    #             if original_observation is not None:
    #                 start_observations = original_observation
    #                 start_actions = original_action
    #                 original_observation = None
    #             else:
    #                 start_observations = torch.from_numpy(dataset._observations[start_indexes.cpu().numpy()]).to(self._impl.device)
    #                 with torch.no_grad():
    #                     start_actions = self._network._impl._policy(start_observations)

    #             mus, logstds = []
    #             for model in self._impl._dynamics._models:
    #                 mu, logstd = self._impl._dynamics.compute_stats(start_observations, start_actions)
    #                 mus.append(mu)
    #                 logstds.append(logstd)
    #             mus = mus.stack(dim=1)
    #             logstds = logstds.stack(dim=1)
    #             mus = mus[torch.arange(start_observations.shape[0]), torch.randint(len(self._impl._dynamics._models), size=(start_observations.shape[0],))]
    #             logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(self._models), size=(start_observations.shape[0],))]

    #             near_indexes, _, _ = similar_mb(mus, logstds, dataset._observations, self._impl._dynamics, topk=self._topk)
    #             near_indexes = near_indexes.reshape((near_indexes.shape[0] * near_indexes.shape[1]))
    #             near_indexes = torch.unique(near_indexes).cpu().numpy()
    #             start_indexes = near_indexes
    #             for replay_index in replay_indexes:
    #                 start_indexes = np.setdiff1d(start_indexes, replay_index, True)
    #             start_indexes = start_indexes[start_indexes != dataset._observations.shape[0] - 1]
    #             start_next_indexes = start_indexes + 1

    #             for i in range(start_observations.shape[0]):
    #                 transition = Transition(
    #                     observation_shape = self._impl.observation_shape,
    #                     action_size = self._impl.action_size,
    #                     observation = dataset._observations[start_indexes[i]],
    #                     action = dataset._actions[start_indexes[i]],
    #                     reward = dataset._rewards[start_indexes[i]],
    #                     next_observation = dataset._observations[start_next_indexes[i]],
    #                     next_action = dataset._actions[start_next_indexes[i]],
    #                     next_reward = dataset._rewards[start_next_indexes[i]],
    #                     terminal = dataset._terminals[start_indexes[i]]
    #                 )
    #                 new_transitions.append(transition)

    #             start_rewards = dataset._rewards[start_indexes]
    #             if max_reward is not None:
    #                 start_indexes = start_indexes[start_rewards >= max_reward]
    #             start_terminals = dataset._terminals[start_indexes]
    #             start_indexes = start_indexes[start_terminals != 1]
    #             if start_indexes.shape[0] == 0:
    #                 break
    #             replay_indexes.append(start_indexes)
    #             replay_indexes = np.concatenate(replay_indexes, dim=0)
    #         return new_transitions
    #     else:
    #         return None
