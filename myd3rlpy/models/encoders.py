import copy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

from torch import nn

from d3rlpy.models.encoders import EncoderFactory, DefaultEncoderFactory, register_encoder_factory, _create_activation
from myd3rlpy.models.torch.encoders import EnsembleEncoder, EnsembleEncoderWithAction
# from myd3rlpy.models.torch import (
#     VAEEncoder,
#     VAEEncoderWithAction,
# )
# 
# 
# class VAEEncoderFactory(EncoderFactory):
#     """Vector encoder factory class.
# 
#     This is the default encoder factory for vector observation.
# 
#     Args:
#         hidden_units (list): list of hidden unit sizes. If ``None``, the
#             standard architecture with ``[256, 256]`` is used.
#         activation (str): activation function name.
#         use_batch_norm (bool): flag to insert batch normalization layers.
#         use_dense (bool): flag to use DenseNet architecture.
#         dropout_rate (float): dropout probability.
# 
#     """
# 
#     TYPE: ClassVar[str] = "vector"
#     _hidden_units: Sequence[int]
#     _activation: str
#     _use_batch_norm: bool
#     _dropout_rate: Optional[float]
#     _use_dense: bool
# 
#     def __init__(
#         self,
#         hidden_units1: Optional[Sequence[int]] = None,
#         hidden_units2: Optional[Sequence[int]] = None,
#         activation: str = "relu",
#         use_batch_norm: bool = False,
#         dropout_rate: Optional[float] = None,
#         use_dense: bool = False,
#     ):
#         if hidden_units1 is None:
#             self._hidden_units1 = [256, 256]
#         else:
#             self._hidden_units1 = hidden_units1
#         if hidden_units2 is None:
#             self._hidden_units2 = [256, 256]
#         else:
#             self._hidden_units2 = hidden_units2
#         self._activation = activation
#         self._use_batch_norm = use_batch_norm
#         self._dropout_rate = dropout_rate
#         self._use_dense = use_dense
# 
#     def create(self, observation_shape: Sequence[int]) -> VAEEncoder:
#         assert len(observation_shape) == 1
#         return VAEEncoder(
#             observation_shape=observation_shape,
#             hidden_units1=self._hidden_units1,
#             hidden_units2=self._hidden_units2,
#             use_batch_norm=self._use_batch_norm,
#             dropout_rate=self._dropout_rate,
#             use_dense=self._use_dense,
#             activation=_create_activation(self._activation),
#         )
# 
#     def create_with_action(
#         self,
#         observation_shape: Sequence[int],
#         action_size: int,
#         discrete_action: bool = False,
#     ) -> VAEEncoderWithAction:
#         assert len(observation_shape) == 1
#         return VAEEncoderWithAction(
#             observation_shape=observation_shape,
#             action_size=action_size,
#             hidden_units1=self._hidden_units1,
#             hidden_units2=self._hidden_units2,
#             use_batch_norm=self._use_batch_norm,
#             dropout_rate=self._dropout_rate,
#             use_dense=self._use_dense,
#             discrete_action=discrete_action,
#             activation=_create_activation(self._activation),
#         )
# 
#     def get_params(self, deep: bool = False) -> Dict[str, Any]:
#         if deep:
#             hidden_units1 = copy.deepcopy(self._hidden_units1)
#         else:
#             hidden_units1 = self._hidden_units1
#         if deep:
#             hidden_units2 = copy.deepcopy(self._hidden_units2)
#         else:
#             hidden_units2 = self._hidden_units2
#         params = {
#             "hidden_units1": hidden_units1,
#             "hidden_units2": hidden_units2,
#             "activation": self._activation,
#             "use_batch_norm": self._use_batch_norm,
#             "dropout_rate": self._dropout_rate,
#             "use_dense": self._use_dense,
#         }
#         return params

class EnsembelDefaultEncoderFactory(EncoderFactory):
    TYPE: ClassVar[str] = "ensemble_default"
    def __init__(self, activation="relu", use_batch_norm=False, dropout_rate=None, n_ensemble=10):
        self._default_encoder_factories = [DefaultEncoderFactory(activation, use_batch_norm, dropout_rate) for _ in range(n_ensemble)]
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._n_ensemble = n_ensemble
    def create(self, observation_shape):
        default_encoders = [default_encoder_factory.create(observation_shape) for default_encoder_factory in self._default_encoder_factories]
        ensemble_default_encoder = EnsembleEncoder(default_encoders)
        return ensemble_default_encoder
    def create_with_action(self, observation_shape, action_size, discrete_action):
        default_encoders = [default_encoder_factory.create_with_action(observation_shape, action_size, discrete_action) for default_encoder_factory in self._default_encoder_factories]
        ensemble_default_encoder = EnsembleEncoder(default_encoders)
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        params = {
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
            "n_ensemble": self._n_ensemble
        }
        return params

# register_encoder_factory(VAEEncoderFactory)
register_encoder_factory(EnsembelDefaultEncoderFactory)
