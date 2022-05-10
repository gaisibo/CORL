import copy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

from torch import nn

from d3rlpy.decorators import pretty_repr
from d3rlpy.torch_utility import Swish
from d3rlpy.models.encoders import EncoderFactory, register_encoder_factory, _create_activation
from d3rlpy.models.torch import (
    Encoder,
    EncoderWithAction,
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)


class LargeVectorEncoderFactory(EncoderFactory):
    """Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): list of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_dense (bool): flag to use DenseNet architecture.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "large_vector"
    _hidden_units: Sequence[int]
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool

    def __init__(
        self,
        hidden_units: Optional[Sequence[int]] = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
    ):
        if hidden_units is None:
            self._hidden_units = [1024, 1024]
        else:
            self._hidden_units = hidden_units
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._use_dense = use_dense

    def create(self, observation_shape: Sequence[int]) -> VectorEncoder:
        assert len(observation_shape) == 1
        return VectorEncoder(
            observation_shape=observation_shape,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            activation=_create_activation(self._activation),
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        assert len(observation_shape) == 1
        return VectorEncoderWithAction(
            observation_shape=observation_shape,
            action_size=action_size,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            discrete_action=discrete_action,
            activation=_create_activation(self._activation),
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            hidden_units = copy.deepcopy(self._hidden_units)
        else:
            hidden_units = self._hidden_units
        params = {
            "hidden_units": hidden_units,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
            "use_dense": self._use_dense,
        }
        return params

register_encoder_factory(LargeVectorEncoderFactory)
