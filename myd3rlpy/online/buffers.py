from abc import ABCMeta
from typing import List

import numpy as np

from myd3rlpy.dataset import (
    MDPDataset,
    Transition
)
from d3rlpy.online.buffers import _Buffer as Old_Buffer, Buffer as OldBuffer, ReplayBuffer as OldReplayBuffer, BasicSampleMixin as OldBasicSampleMixin


class _Buffer(_Buffer, metaclass=ABCMeta)

    def to_mdp_dataset(self) -> MDPDataset:
        """Convert replay data into static dataset.

        The length of the dataset can be longer than the length of the replay
        buffer because this conversion is done by tracing ``Transition``
        objects.

        Returns:
            MDPDataset object.

        """
        # get the last transitions
        tail_transitions: List[Transition] = []
        for transition in self._transitions:
            if transition.next_transition is None:
                tail_transitions.append(transition)

        observations = []
        actions = []
        rewards = []
        terminals = []
        episode_terminals = []
        for transition in tail_transitions:

            # trace transition to the beginning
            episode_transitions: List[Transition] = []
            while True:
                episode_transitions.append(transition)
                if transition.prev_transition is None:
                    break
                transition = transition.prev_transition
            episode_transitions.reverse()

            # stack data
            for i, episode_transition in enumerate(episode_transitions):
                observations.append(episode_transition.observation)
                actions.append(episode_transition.action)
                rewards.append(episode_transition.reward)
                terminals.append(episode_transition.terminal)
                episode_terminals.append(i == len(episode_transitions) - 1)

        if len(self._observation_shape) == 3:
            observations = np.asarray(observations, dtype=np.uint8)
        else:
            observations = np.asarray(observations, dtype=np.float32)

        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
        )

class Buffer(_Buffer, OldBuffer):
    pass
class BasicSampleMixin(OldBasicSampleMixin):

    def sample(
        self,
        batch_size: int,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> TransitionMiniBatch:
        indices = np.random.choice(len(self._transitions), batch_size)
        transitions = [self._transitions[index] for index in indices]
        batch = TransitionMiniBatch(transitions, n_frames, n_steps, gamma)
        return batch

class ReplayBuffer(BasicSampleMixin, Buffer, ReplayBuffer):
