from typing import Sequence, cast, Optional

import torch
from torch import nn

from d3rlpy.models.encoders import EncoderFactory
from myd3rlpy.models.torch.siamese import Phi, Psi

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
# 
# def create_continuous_q_function(
#     observation_shape: Sequence[int],
#     action_size: int,
#     encoder_factory: EncoderFactory,
#     q_func_factory: QFunctionFactory,
#     n_ensembles: int = 1,
# ) -> EnsembleContinuousQFunction:
#     if q_func_factory.share_encoder:
#         encoder = encoder_factory.create_with_action(
#             observation_shape, action_size
#         )
#         # normalize gradient scale by ensemble size
#         for p in cast(nn.Module, encoder).parameters():
#             p.register_hook(lambda grad: grad / n_ensembles)
# 
#     q_funcs = []
#     for _ in range(n_ensembles):
#         if not q_func_factory.share_encoder:
#             encoder = encoder_factory.create_with_action(
#                 observation_shape, action_size
#             )
#         q_funcs.append(q_func_factory.create_continuous(encoder))
#     return EnsembleContinuousQFunction(q_funcs)
