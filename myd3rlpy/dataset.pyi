# pylint: disable=multiple-statements

from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from d3rlpy.dataset imort Transition as OldTransition, TransitionMiniBatch as OldTransitionMiniBatch, Episode as OldEpisode, MDPDataset as OldMDPDataset

class Transition(OldTransition):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        observation: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        rtg: float,
        next_observation: np.ndarray,
        next_action: Union[int, np.ndarray],
        terminal: float,
        prev_transition: Optional["Transition"] = ...,
        next_transition: Optional["Transition"] = ...,
    ): ...
    @property
    def rtg(self) -> float: ...

class TransitionMiniBatch(OldTransitionMiniBatch):
    @property
    def rtgs(self) -> np.ndarray: ...

class Episode(OldEpisode):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        rtgs: np.ndarray,
        terminal: bool = ...,
    ): ...
    @property
    def rtgs(self) -> np.ndarray: ...

class MDPDataset(OldMDPDataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        rtgs: np.ndarray,
        terminals: np.ndarray,
        episode_terminals: Optional[np.ndarray] = ...,
        discrete_action: Optional[bool] = ...,
    ): ...
    @property
    def rtgs(self) -> np.ndarray: ...
    def clip_rtgs(
        self, low: Optional[float], high: Optional[float]
    ) -> None: ...
    def append(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        rtgs: np.ndarray,
        terminals: np.ndarray,
        episode_terminals: Optional[np.ndarray] = ...,
    ) -> None: ...
