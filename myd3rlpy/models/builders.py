from typing import Sequence, cast, Optional

import torch
from torch import nn

from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import EnsembleContinuousQFunction
from myd3rlpy.models.torch.q_functions.ensemble_q_function import ParallelEnsembleContinuousQFunction
from myd3rlpy.models.torch.v_functions import EnsembleValueFunction
from myd3rlpy.models.torch.siamese import Phi, Psi
from myd3rlpy.models.torch.embed import Embed

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

def create_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    reduction: str = 'mean',
    n_ensembles: int = 1,
) -> EnsembleContinuousQFunction:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create_with_action(
            observation_shape, action_size
        )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
        q_funcs.append(q_func_factory.create_continuous(encoder))
    return EnsembleContinuousQFunction(q_funcs, reduction=reduction)

def create_parallel_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    reduction: str = 'mean',
    n_ensembles: int = 1,
) -> ParallelEnsembleContinuousQFunction:
    return ParallelEnsembleContinuousQFunction(ensemble_size=n_ensembles, hidden_sizes=[256, 256], input_size=observation_shape[0] + action_size, output_size=1, reduction=reduction)

def create_value_function(
    observation_shape: Sequence[int], encoder_factory: EncoderFactory
) -> EnsembleValueFunction:
    encoder = encoder_factory.create(observation_shape)
    return EnsembleValueFunction(encoder)

def create_embed(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
    output_dims: int,
) -> Phi:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    return Embed(encoder, output_dims=output_dims)
