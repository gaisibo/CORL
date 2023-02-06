import copy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

from torch import nn

from d3rlpy.models.encoders import _create_activation
from myd3rlpy.models.torch.vaes import VAE, VectorVAE


class VAEFactory:
    TYPE: ClassVar[str] = "none"

    def create(self, observation_shape: Sequence[int]) -> VAE:
        """Returns PyTorch's state enocder module.

        Args:
            observation_shape: observation shape.

        Returns:
            an enocder object.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns encoder type.

        Returns:
            encoder type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns encoder parameters.

        Args:
            deep: flag to deeply copy the parameters.

        Returns:
            encoder parameters.

        """
        raise NotImplementedError

class VectorVAEFactory(VAEFactory):
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

    TYPE: ClassVar[str] = "vector"
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
            self._hidden_units = [256, 256]
        else:
            self._hidden_units = hidden_units
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._use_dense = use_dense

    def create(self, observation_shape: Sequence[int], feature_size: int) -> VAE:
        assert len(observation_shape) == 1
        return VectorVAE(
            observation_shape=observation_shape,
            feature_size=feature_size,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
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

VAE_LIST: Dict[str, Type[VAEFactory]] = {}


def register_vae_factory(cls: Type[VAEFactory]) -> None:
    """Registers encoder factory class.

    Args:
        cls: encoder factory class inheriting ``EncoderFactory``.

    """
    is_registered = cls.TYPE in VAE_LIST
    assert not is_registered, f"{cls.TYPE} seems to be already registered"
    VAE_LIST[cls.TYPE] = cls

def create_vae_factory(name: str, **kwargs: Any) -> VAEFactory:
    """Returns registered encoder factory object.

    Args:
        name: regsitered encoder factory type name.
        kwargs: encoder arguments.

    Returns:
        encoder factory object.

    """
    assert name in VAE_LIST, f"{name} seems not to be registered."
    factory = VAE_LIST[name](**kwargs)
    assert isinstance(factory, VAEFactory)
    return factory
