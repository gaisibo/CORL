from typing import Any, Dict, Optional, Sequence

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.algos.base import AlgoBase
from myd3rlpy.algos.torch.siamese_impl import SiameseImpl


class Siamese(AlgoBase):
    r"""Deep Deterministic Policy Gradients algorithm.
    DDPG is an actor-critic algorithm that trains a Q function parametrized
    with :math:`\theta` and a policy function parametrized with :math:`\phi`.
    .. math::
        L(\theta) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D} \Big[(r_{t+1}
            + \gamma Q_{\theta'}\big(s_{t+1}, \pi_{\phi'}(s_{t+1}))
            - Q_\theta(s_t, a_t)\big)^2\Big]
    .. math::
        J(\phi) = \mathbb{E}_{s_t \sim D} \Big[Q_\theta\big(s_t, \pi_\phi(s_t)\big)\Big]
    where :math:`\theta'` and :math:`\phi` are the target network parameters.
    There target network parameters are updated every iteration.
    .. math::
        \theta' \gets \tau \theta + (1 - \tau) \theta'
        \phi' \gets \tau \phi + (1 - \tau) \phi'
    References:
        * `Silver et al., Deterministic policy gradient algorithms.
          <http://proceedings.mlr.press/v32/silver14.html>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_
    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q function.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.ddpg_impl.DDPGImpl): algorithm implementation.
    """

    _phi_learning_rate: float
    _psi_learning_rate: float
    _phi_optim_factory: OptimizerFactory
    _psi_optim_factory: OptimizerFactory
    _phi_encoder_factory: EncoderFactory
    _psi_encoder_factory: EncoderFactory
    _target_reduction_type: str
    _use_gpu: Optional[Device]
    _impl: Optional[SiameseImpl]

    def __init__(
        self,
        policy: AlgoBase,
        *,
        phi_learning_rate: float = 1e-3,
        psi_learning_rate: float = 1e-3,
        phi_optim_factory: OptimizerFactory = AdamFactory(),
        psi_optim_factory: OptimizerFactory = AdamFactory(),
        phi_encoder_factory: EncoderArg = "default",
        psi_encoder_factory: EncoderArg = "default",
        batch_size: int = 256 * 4,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_sample_actions = 256,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[SiameseImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory
        self._phi_encoder_factory = check_encoder(phi_encoder_factory)
        self._psi_encoder_factory = check_encoder(psi_encoder_factory)
        self._policy = policy
        self._n_sample_actions = n_sample_actions
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(self, observation_shape: Sequence[int], action_size: int) -> None:
        self._impl = SiameseImpl(
            policy=self._policy,
            observation_shape=observation_shape,
            action_size=action_size,
            phi_learning_rate=self._phi_learning_rate,
            psi_learning_rate=self._psi_learning_rate,
            phi_optim_factory=self._phi_optim_factory,
            psi_optim_factory=self._psi_optim_factory,
            phi_encoder_factory=self._phi_encoder_factory,
            psi_encoder_factory=self._psi_encoder_factory,
            gamma=self._gamma,
            n_sample_actions=self._n_sample_actions,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        quarter_batch_len = len(batch) // 4
        phi_batch1 = TransitionMiniBatch(batch[: quarter_batch_len])
        phi_batch2 = TransitionMiniBatch(batch[quarter_batch_len : quarter_batch_len * 2])
        psi_batch1 = TransitionMiniBatch(batch[quarter_batch_len * 2 : quarter_batch_len * 3])
        psi_batch2 = TransitionMiniBatch(batch[quarter_batch_len * 3 :])
        phi_loss = self._impl.update_phi(phi_batch1, phi_batch2)
        psi_loss = self._impl.update_psi(psi_batch1, psi_batch2)

        return {"phi_loss": phi_loss, "psi_loss": psi_loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
