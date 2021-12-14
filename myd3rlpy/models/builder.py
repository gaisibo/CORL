from typing import Sequence, cast, Optional

import torch
from torch import nn

from myd3rlpy.models.torch.siamese import Phi, Psi
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.q_functions.ensemble_q_function import EnsembleContinuousQFunction
from d3rlpy.models.torch.policies import DeterministicPolicy

def create_phi(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    task_id_size: Optional[int] = None,
) -> Phi:
    if task_id_size is None:
        encoder = encoder_factory.create_with_action(observation_shape, action_size)
    else:
        encoder = encoder_factory.create_with_action_with_task_id(observation_shape, action_size, task_id_size)
    return Phi(encoder)

def create_psi(
    observation_shape: Sequence[int],
    encoder_factory: EncoderFactory,
    task_id_size: Optional[int] = None,
) -> Psi:
    if task_id_size is None:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = encoder_factory.create_with_task_id(observation_shape, task_id_size)
    return Psi(encoder)

def create_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
    task_id_size: Optional[int] = None,
) -> EnsembleContinuousQFunction:
    if q_func_factory.share_encoder:
        if task_id_size is not None:
            encoder = encoder_factory.create_with_action_with_task_id(
                observation_shape, action_size, task_id_size
            )
        else:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            if task_id_size is not None:
                encoder = encoder_factory.create_with_action_with_task_id(
                    observation_shape, action_size, task_id_size
                )
            else:
                encoder = encoder_factory.create_with_action(
                    observation_shape, action_size
                )
        q_funcs.append(q_func_factory.create_continuous(encoder))
    return EnsembleContinuousQFunction(
        q_funcs, bootstrap=q_func_factory.bootstrap
    )


def create_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    task_id_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicPolicy:
    encoder = encoder_factory.create_with_task_id(observation_shape, task_id_size)
    return DeterministicPolicy(encoder, action_size)
