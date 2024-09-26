from typing import Dict, Optional, List, Sequence
from torch import Tensor

from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR

from myd3rlpy.algos.o2o import O2OBase
from myd3rlpy.algos.st_td3 import STTD3
from myd3rlpy.algos.torch.st_impl import STImpl

class O2OTD3(O2OBase, STTD3):
    def _update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}

        critic_loss, replay_critic_loss = self._impl.update_critic(value_batch, None, online=online)
        metrics.update({"critic_loss": critic_loss})
        metrics.update({"replay_critic_loss": replay_critic_loss})

        if self._grad_step % self._update_actor_interval == 0:
            actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, online=online)
            metrics.update({"actor_loss": actor_loss})
            metrics.update({"replay_actor_loss": replay_actor_loss})
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        _impl_dict = dict(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            critic_replay_type=self._critic_replay_type,
            critic_replay_lambda=self._critic_replay_lambda,
            actor_replay_type=self._actor_replay_type,
            actor_replay_lambda=self._actor_replay_lambda,
            # conservative_threshold=self._conservative_threshold,
            gamma=self._gamma,
            gem_alpha=self._gem_alpha,
            agem_alpha=self._agem_alpha,
            ewc_rwalk_alpha=self._ewc_rwalk_alpha,
            damping=self._damping,
            epsilon=self._epsilon,
            tau=self._tau,
            n_critics=self._n_critics,
            target_smoothing_sigma=self._target_smoothing_sigma,
            target_smoothing_clip=self._target_smoothing_clip,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            fine_tuned_step = self._fine_tuned_step,
        )
        if self._impl_name == 'td3_plus_bc':
            from myd3rlpy.algos.torch.o2o_td3_plus_bc_impl import O2OTD3PlusBCImpl as O2OImpl
            _impl_dict['alpha'] = self._alpha
        elif self._impl_name == 'td3':
            from myd3rlpy.algos.torch.o2o_td3_impl import O2OTD3Impl as O2OImpl
        else:
            print(self._impl_name)
            raise NotImplementedError
        self._impl = O2OImpl(**_impl_dict)
        self._impl.build()
