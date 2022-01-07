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
)

import gym
import numpy as np
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
from d3rlpy.dataset import Episode, MDPDataset
from d3rlpy.iterators import RandomIterator, RoundIterator, TransitionIterator
from d3rlpy.logger import LOG
from d3rlpy.base import LearnableBase
from d3rlpy.argument_utility import ActionScalerArg, ScalerArg
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics
from d3rlpy.dynamics.torch.probabilistic_ensemble_dynamics_impl import ProbabilisticEnsembleDynamicsImpl


class ProbabilisticEnsembleDynamics(ProbabilisticEnsembleDynamics):

    """ProbabilisticEnsembleDynamics with following sample"""

    def __init__(
        self,
        *,
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
        self._topk = topk
        self._network = network

    def generate_new_data(self, dataset, original, in_task=False, max_export_time = 0, max_reward=None):
        # 关键算法
        original_observation = torch.cat([original, torch.from_numpy(task_id_numpy).to(torch.float32).to(self._impl.device)], dim=1)
        original_action = self._network._impl._policy(original_observation)
        replay_indexes = []
        new_transitions = []

        export_time = 0
        start_indexes = torch.zeros(0)
        while start_indexes.shape[0] != 0 and original_observation is not None and export_time < max_export_time:
            if original_observation is not None:
                start_observations = original_observation
                start_actions = original_action
                original_observation = None
            else:
                start_observations = torch.from_numpy(dataset._observations[start_indexes.cpu().numpy()]).to(self._impl.device)
                start_actions = self._network._impl._policy(start_observations)

            mus, logstds = []
            for model in self._impl._dynamics._models:
                mu, logstd = self._impl._dynamics.compute_stats(start_observations, start_actions)
                mus.append(mu)
                logstds.append(logstd)
            mus = mus.stack(dim=1)
            logstds = logstds.stack(dim=1)
            mus = mus[torch.arange(start_observations.shape[0]), torch.randint(len(self._impl._dynamics._models), size=(start_observations.shape[0],))]
            logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(self._models), size=(start_observations.shape[0],))]

            near_indexes, _, _ = similar_mb(mus, logstds, dataset._observations, self._impl._dynamics, topk=self._topk)
            near_indexes = near_indexes.reshape((near_indexes.shape[0] * near_indexes.shape[1]))
            near_indexes = torch.unique(near_indexes).cpu().numpy()
            start_indexes = near_indexes
            for replay_index in replay_indexes:
                start_indexes = np.setdiff1d(start_indexes, replay_index)
            start_next_indexes = np.where(start_indexes + 1 < dataset._observations.shape[0], start_indexes + 1, 0)

            for i in range(start_observations.shape[0]):
                transition = Transition(
                    observation_shape = self._impl.observation_shape,
                    action_size = self._impl.action_size,
                    observation = dataset._observations[start_indexes],
                    action = dataset._actions[start_indexes],
                    reward = dataset._rewards[start_indexes],
                    next_observation = dataset._observations[start_next_indexes],
                    next_action = dataset._actions[start_next_indexes],
                    next_reward = dataset._rewards[start_next_indexes],
                    terminal = dataset._terminals[start_indexes]
                )
                new_transitions.append(transition)

            start_rewards = dataset._rewards[start_indexes]
            if max_reward is not None:
                start_indexes = start_indexes[start_rewards >= max_reward]
            start_terminals = dataset._terminals[start_indexes]
            start_indexes = start_indexes[start_terminals != 1]
            if start_indexes.shape[0] == 0:
                break
            replay_indexes.append(start_indexes)
            replay_indexes = np.concatenate(replay_indexes, dim=0)
        return new_transitions
