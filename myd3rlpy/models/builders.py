from typing import Sequence, cast, Optional

import torch
from torch import nn

from d3rlpy.models.encoders import EncoderFactory
from myd3rlpy.models.torch.siamese import Phi, Psi
from myd3rlpy.models.torch.dynamics import ProbabilisticEnsembleDynamicsWithLogStdModel, ProbabilisticDynamicsWithLogStdModel

def create_phi(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> Phi:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    return Phi(encoder)

def create_psi(
    observation_shape: Sequence[int],
    encoder_factory: EncoderFactory,
) -> Psi:
    encoder = encoder_factory.create(observation_shape)
    return Psi(encoder)
