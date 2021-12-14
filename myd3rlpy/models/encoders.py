import copy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

from torch import nn

from d3rlpy.decorators import pretty_repr
from d3rlpy.torch_utility import Swish
from myd3rlpy.models.torch import (
    EncoderWithTaskID,
    EncoderWithActionWithTaskID,
    PixelEncoderWithTaskID,
    PixelEncoderWithActionWithTaskID,
    VectorEncoderWithTaskID,
    VectorEncoderWithActionWithTaskID,
)
from d3rlpy.models.encoders import _create_activation, EncoderFactory, PixelEncoderFactory as PixelEncoderFactoryO, VectorEncoderFactory as VectorEncoderFactoryO, DefaultEncoderFactory as DefaultEncoderFactoryO, DenseEncoderFactory as DenseEncoderFactoryO, register_encoder_factory, create_encoder_factory


class PixelEncoderFactory(PixelEncoderFactoryO):
    """Pixel encoder factory class.
    This is the default encoder factory for image observation.
    Args:
        filters (list): list of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): the last linear layer size.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.
    """
    def create_with_task_id(self, observation_shape: Sequence[int], task_id_size: int, discrete_task_id: bool = False) -> PixelEncoderWithTaskID:
        assert len(observation_shape) == 3
        return PixelEncoderWithTaskID(
            observation_shape=observation_shape,
            task_id_size=task_id_size,
            filters=self._filters,
            feature_size=self._feature_size,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            activation=_create_activation(self._activation),
            discrete_task_id=discrete_task_id,
        )

    def create_with_action_with_task_id(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        task_id_size: int,
        discrete_action: bool = False,
        discrete_task_id: bool = False,
    ) -> PixelEncoderWithActionWithTaskID:
        assert len(observation_shape) == 3
        return PixelEncoderWithActionWithTaskID(
            observation_shape=observation_shape,
            action_size=action_size,
            task_id_size=task_id_size,
            filters=self._filters,
            feature_size=self._feature_size,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            discrete_action=discrete_action,
            discrete_task_id=discrete_task_id,
            activation=_create_activation(self._activation),
        )


class VectorEncoderFactory(VectorEncoderFactoryO):
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
    def create_with_task_id(self, observation_shape: Sequence[int], task_id_size: int, discrete_task_id: bool = False) -> VectorEncoderWithTaskID:
        assert len(observation_shape) == 1
        return VectorEncoderWithTaskID(
            observation_shape=observation_shape,
            task_id_size=task_id_size,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            activation=_create_activation(self._activation),
            discrete_task_id=discrete_task_id,
        )

    def create_with_action_with_task_id(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        task_id_size: int,
        discrete_action: bool = False,
        discrete_task_id: bool = False,
    ) -> VectorEncoderWithActionWithTaskID:
        assert len(observation_shape) == 1
        return VectorEncoderWithActionWithTaskID(
            observation_shape=observation_shape,
            action_size=action_size,
            task_id_size=task_id_size,
            hidden_units=self._hidden_units,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            discrete_action=discrete_action,
            discrete_task_id=discrete_task_id,
            activation=_create_activation(self._activation),
        )


class DefaultEncoderFactory(DefaultEncoderFactoryO):
    """Default encoder factory class.
    This encoder factory returns an encoder based on observation shape.
    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.
    """
    def create_with_task_id(self, observation_shape: Sequence[int], task_id_size: int, discrete_task_id: bool = False) -> EncoderWithTaskID:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        return factory.create(observation_shape, task_id_size, discrete_task_id)

    def create_with_action_with_task_id(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        task_id_size: int,
        discrete_action: bool = False,
        discrete_task_id: bool = False,
    ) -> EncoderWithAction:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        return factory.create_with_action_with_task_id(
            observation_shape, action_size, task_id_size, discrete_action, discrete_task_id
        )


class DenseEncoderFactory(DenseEncoderFactoryO):
    """DenseNet encoder factory class.
    This is an alias for DenseNet architecture proposed in D2RL.
    This class does exactly same as follows.
    .. code-block:: python
       from d3rlpy.encoders import VectorEncoderFactory
       factory = VectorEncoderFactory(hidden_units=[256, 256, 256, 256],
                                      use_dense=True)
    For now, this only supports vector observations.
    References:
        * `Sinha et al., D2RL: Deep Dense Architectures in Reinforcement
          Learning. <https://arxiv.org/abs/2010.09163>`_
    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.
    """
    def create_with_task_id(self, observation_shape: Sequence[int], task_id_size: int, discrete_task_id: bool = False) -> VectorEncoder:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self._activation,
            use_dense=True,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
        )
        return factory.create(observation_shape, task_id_size, discrete_task_id)

    def create_with_action_with_task_id(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        task_id_size: int,
        discrete_action: bool = False,
        discrete_task_id: bool = False,
    ) -> VectorEncoderWithAction:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self._activation,
            use_dense=True,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
        )
        return factory.create_with_action_with_task_id(
            observation_shape, action_size, task_id_size, discrete_action, discrete_task_id
        )

ENCODER_LIST: Dict[str, Type[EncoderFactory]] = {}
register_encoder_factory(VectorEncoderFactory)
register_encoder_factory(PixelEncoderFactory)
register_encoder_factory(DefaultEncoderFactory)
register_encoder_factory(DenseEncoderFactory)
