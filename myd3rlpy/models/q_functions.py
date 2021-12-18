from typing import Any, ClassVar, Dict, Type

from d3rlpy.decorators import pretty_repr
from d3rlpy.models.q_functions import QFunctionFactory, MeanQFunctionFactory, QRQFunctionFactory, IQNQFunctionFactory, FQFQFunctionFactory, register_q_func_factory
from myd3rlpy.models.torch.q_functions.ensemble_q_function import EnsembleContinuousQFunctionWithTaskID, EnsembleDiscreteQFunctionWithTaskID
from myd3rlpy.models.torch.q_functions.qr_q_function import ContinuousQRQFunctionWithTaskID, DiscreteQRQFunctionWithTaskID
from myd3rlpy.models.torch.q_functions.fqf_q_function import ContinuousFQFQFunctionWithTaskID, DiscreteFQFQFunctionWithTaskID
from myd3rlpy.models.torch.q_functions.fqf_q_function import ContinuousFQFQFunctionWithTaskID, DiscreteFQFQFunctionWithTaskID
from myd3rlpy.models.torch.q_functions.mean_q_function import ContinuousMeanQFunctionWithTaskID, DiscreteMeanQFunctionWithTaskID
from myd3rlpy.models.torch.q_functions.iqn_q_function import ContinuousIQNQFunctionWithTaskID, DiscreteIQNQFunctionWithTaskID
from myd3rlpy.models.torch.encoders import EncoderWithTaskID, EncoderWithActionWithTaskID


@pretty_repr
class QFunctionFactoryWithTaskID(QFunctionFactory):
    TYPE: ClassVar[str] = "noneid"

    _bootstrap: bool
    _share_encoder: bool

    def __init__(self, bootstrap: bool, share_encoder: bool):
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder

    def create_discrete_with_task_id(
        self, encoder: EncoderWithTaskID, action_size: int, task_id_size: int
    ):
        """Returns PyTorch's Q function module.
        Args:
            encoder: an encoder module that processes the observation to
                obtain feature representations.
            action_size: dimension of discrete action-space.
        Returns:
            discrete Q function object.
        """
        raise NotImplementedError

    def create_continuous_with_task_id(
        self, encoder: EncoderWithActionWithTaskID
    ):
        """Returns PyTorch's Q function module.
        Args:
            encoder: an encoder module that processes the observation and
                action to obtain feature representations.
        Returns:
            continuous Q function object.
        """
        raise NotImplementedError


class MeanQFunctionFactoryWithTaskID(MeanQFunctionFactory):
    """Standard Q function factory class.
    This is the standard Q function factory class.
    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_
    Args:
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder over multiple Q functions.
    """
    TYPE: ClassVar[str] = "meanid"
    def create_discrete_with_task_id(
        self,
        encoder: EncoderWithTaskID,
        action_size: int,
        task_id_size: int,
    ) -> DiscreteMeanQFunctionWithTaskID:
        return DiscreteMeanQFunctionWithTaskID(encoder, action_size, task_id_size)

    def create_continuous_with_task_id(
        self,
        encoder: EncoderWithActionWithTaskID,
    ) -> ContinuousMeanQFunctionWithTaskID:
        return ContinuousMeanQFunctionWithTaskID(encoder)


class QRQFunctionFactoryWithTaskID(QRQFunctionFactory):
    """Quantile Regression Q function factory class.
    References:
        * `Dabney et al., Distributional reinforcement learning with quantile
          regression. <https://arxiv.org/abs/1710.10044>`_
    Args:
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
    """
    TYPE: ClassVar[str] = "qrqid"
    def create_discrete_with_task_id(
            self, encoder: EncoderWithTaskID, action_size: int, task_id_size: int,
    ) -> DiscreteQRQFunctionWithTaskID:
        return DiscreteQRQFunctionWithTaskID(encoder, action_size, task_id_size, self._n_quantiles)

    def create_continuous_with_task_id(
        self,
        encoder: EncoderWithActionWithTaskID,
    ) -> ContinuousQRQFunctionWithTaskID:
        return ContinuousQRQFunctionWithTaskID(encoder, self._n_quantiles)


class IQNQFunctionFactoryWithTaskID(IQNQFunctionFactory):
    """Implicit Quantile Network Q function factory class.
    References:
        * `Dabney et al., Implicit quantile networks for distributional
          reinforcement learning. <https://arxiv.org/abs/1806.06923>`_
    Args:
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
        n_greedy_quantiles: the number of quantiles for inference.
        embed_size: the embedding size.
    """
    TYPE: ClassVar[str] = "iqnid"
    def create_discrete_with_task_id(
        self,
        encoder: EncoderWithTaskID,
        action_size: int,
        task_size: int,
    ) -> DiscreteIQNQFunctionWithTaskID:
        return DiscreteIQNQFunctionWithTaskID(
            encoder=encoder,
            action_size=action_size,
            task_id_size=task_id_size,
            n_quantiles=self._n_quantiles,
            n_greedy_quantiles=self._n_greedy_quantiles,
            embed_size=self._embed_size,
        )

    def create_continuous_with_task_id(
        self,
        encoder: EncoderWithActionWithTaskID,
    ) -> ContinuousIQNQFunctionWithTaskID:
        return ContinuousIQNQFunctionWithTaskID(
            encoder=encoder,
            n_quantiles=self._n_quantiles,
            n_greedy_quantiles=self._n_greedy_quantiles,
            embed_size=self._embed_size,
        )


class FQFQFunctionFactoryWithTaskID(FQFQFunctionFactory):
    """Fully parameterized Quantile Function Q function factory.
    References:
        * `Yang et al., Fully parameterized quantile function for
          distributional reinforcement learning.
          <https://arxiv.org/abs/1911.02140>`_
    Args:
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
        embed_size: the embedding size.
        entropy_coeff: the coefficiency of entropy penalty term.
    """
    TYPE: ClassVar[str] = "fqfid"
    def create_discrete_with_task_id(
        self,
        encoder: EncoderWithTaskID,
        action_size: int,
        task_size: int,
    ) -> DiscreteFQFQFunctionWithTaskID:
        return DiscreteFQFQFunctionWithTaskID(
            encoder=encoder,
            action_size=action_size,
            task_id_size=task_id_size,
            n_quantiles=self._n_quantiles,
            embed_size=self._embed_size,
            entropy_coeff=self._entropy_coeff,
        )

    def create_continuous_with_task_id(
        self,
        encoder: EncoderWithActionWithTaskID,
    ) -> ContinuousFQFQFunctionWithTaskID:
        return ContinuousFQFQFunctionWithTaskID(
            encoder=encoder,
            n_quantiles=self._n_quantiles,
            embed_size=self._embed_size,
            entropy_coeff=self._entropy_coeff,
        )


register_q_func_factory(MeanQFunctionFactoryWithTaskID)
register_q_func_factory(QRQFunctionFactoryWithTaskID)
register_q_func_factory(IQNQFunctionFactoryWithTaskID)
register_q_func_factory(FQFQFunctionFactoryWithTaskID)
