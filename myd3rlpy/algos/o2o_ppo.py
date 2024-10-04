from typing import Dict, Optional, List, Sequence, Any
from torch import Tensor
from d3rlpy.algos.base import AlgoBase

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory

from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR

from myd3rlpy.algos.o2o import O2OBase
from myd3rlpy.algos.torch.o2o_impl import O2OImpl

class O2OBPPO(O2OBase, AlgoBase):
    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _impl: Optional[O2OImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 1,
        update_critic_target_interval: int = 2,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[O2OBPPOImpl] = None,

        is_clip_decay = True,
        is_lr_decay = True,
        update_critic: bool = False,
        update_critic_interval: int = 10,

        clone_critic: bool = False,
        clone_actor: bool = False,
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
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_critic_target_interval = update_critic_target_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

        self._is_clip_decay = is_clip_decay
        self._is_lr_decay = is_lr_decay
        self._update_critic = update_critic
        self._update_critic_interval = update_critic_interval

        self._clone_critic = clone_critic
        self._clone_actor = clone_actor

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        impl_dict = {
            'observation_shape':observation_shape,
            'action_size':action_size,
            'actor_learning_rate':self._actor_learning_rate,
            'critic_learning_rate':self._critic_learning_rate,
            'temp_learning_rate':self._temp_learning_rate,
            'actor_optim_factory':self._actor_optim_factory,
            'critic_optim_factory':self._critic_optim_factory,
            'temp_optim_factory':self._temp_optim_factory,
            'actor_encoder_factory':self._actor_encoder_factory,
            'critic_encoder_factory':self._critic_encoder_factory,
            'q_func_factory':self._q_func_factory,
            'critic_replay_type':self._critic_replay_type,
            'critic_replay_lambda':self._critic_replay_lambda,
            'actor_replay_type':self._actor_replay_type,
            'actor_replay_lambda':self._actor_replay_lambda,
            'gamma':self._gamma,
            'gem_alpha':self._gem_alpha,
            'agem_alpha':self._agem_alpha,
            'ewc_rwalk_alpha':self._ewc_rwalk_alpha,
            'damping':self._damping,
            'epsilon':self._epsilon,
            'tau':self._tau,
            'n_critics':self._n_critics,
            'initial_temperature':self._initial_temperature,
            'use_gpu':self._use_gpu,
            'scaler':self._scaler,
            'action_scaler':self._action_scaler,
            'reward_scaler':self._reward_scaler,
            'fine_tuned_step': self._fine_tuned_step,

            'is_clip_decay': self._is_clip_decay,
            'is_lr_decay': self._is_lr_decay,
        }
        if self._impl_name == 'ppo':
            from myd3rlpy.algos.torch.o2o_ppo_impl import O2OPPOImpl as O2OImpl
        elif self._impl_name == 'bppo':
            from myd3rlpy.algos.torch.o2o_bppo_impl import O2OBPPOImpl as O2OImpl
        else:
            print(self._impl_name)
            raise NotImplementedError
        self._impl = O2OImpl(
            **impl_dict
        )
        self._impl.build()

    def _update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        if self._grad_step > 200:
            self._impl._is_clip_decay = False
            self._impl._is_lr_decay = False
        if self._update_critic or online:
            critic_loss, replay_critic_loss = 0, 0
            for _ in range(self._update_critic_interval):
                critic_loss, replay_critic_loss = self._impl.update_critic(value_batch, None, clone_critic=self._clone_critic, online=online)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})
        actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, clone_actor=self._clone_actor, online=online)
        metrics.update({"actor_loss": actor_loss})
        metrics.update({"replay_actor_loss": replay_actor_loss})

        return metrics

    def _update_critic_clone(self, value_batch: TransitionMiniBatch) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        critic_loss, value_loss = self._impl.update_critic_clone(value_batch)
        metrics.update({"clone_critic_loss": critic_loss})
        metrics.update({"clone_value_loss": value_loss})
        if self._grad_step % self._update_critic_target_interval == 0:
            self._impl.update_critic_target()
        return metrics

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, real_action_size, batch_size = 64,
    ) -> Optional[List[Transition]]:
        return None
