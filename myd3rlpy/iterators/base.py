from abc import ABCMeta, abstractmethod
from typing import Iterator, List, cast

import numpy as np

from d3rlpy.containers import FIFOQueue
from d3rlpy.iterators.base import TransitionIterator as OldTransitionIterator
from d3rlpy.dataset import TransitionMiniBatch as OldTransitionMiniBatch, Transition as OldTransition
from myd3rlpy.dataset import TransitionMiniBatch, Transition


class TransitionIterator(OldTransitionIterator, metaclass=ABCMeta):

    def __next__(self) -> TransitionMiniBatch:
        if len(self._generated_transitions) > 0:
            real_batch_size = self._real_batch_size
            fake_batch_size = self._batch_size - self._real_batch_size
            transitions = [self.get_next() for _ in range(real_batch_size)]
            transitions += self._sample_generated_transitions(fake_batch_size)
        else:
            transitions = [self.get_next() for _ in range(self._batch_size)]

        if isinstance(transitions[0], Transition):
            batch = TransitionMiniBatch(
                transitions,
                n_frames=self._n_frames,
                n_steps=self._n_steps,
                gamma=self._gamma,
            )
        else:
            batch = OldTransitionMiniBatch(
                transitions,
                n_frames=self._n_frames,
                n_steps=self._n_steps,
                gamma=self._gamma,
            )

        self._count += 1

        return batch
