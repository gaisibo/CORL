from typing import Dict, Optional, List, Sequence
from torch import Tensor

from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR

from myd3rlpy.algos.o2o import O2OBase
from myd3rlpy.algos.st_iql import STIQL
from myd3rlpy.algos.torch.st_impl import STImpl

class O2OIQL(O2OBase, STIQL):
    def _update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        critic_loss, replay_critic_loss = self._impl.update_critic(value_batch, None, clone_critic=self._clone_critic, online=online)
        metrics.update({"critic_loss": critic_loss})
        metrics.update({"replay_critic_loss": replay_critic_loss})
        #print(f"self._impl._q_loss: {self._impl._q_loss}")
        #print(f"self._impl._v_loss: {self._impl._v_loss}")
        #assert False
        if hasattr(self._impl, "_q_loss"):
            metrics.update({"q_loss": self._impl._q_loss.cpu().detach().numpy()})
        if hasattr(self._impl, "_v_loss"):
            metrics.update({"v_loss": self._impl._v_loss.cpu().detach().numpy()})

        actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, clone_actor=self._clone_actor, online=online)
        # actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
        metrics.update({"actor_loss": actor_loss})
        metrics.update({"replay_actor_loss": replay_actor_loss})

        self._impl.update_critic_target()

        return metrics

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int) -> None:
        #print(f"self._actor_replay_type: {self._actor_replay_type}")
        #assert False
        impl_dict = {
            'observation_shape':observation_shape,
            'action_size':action_size,
            'actor_replay_type':self._actor_replay_type,
            'critic_replay_type':self._critic_replay_type,
            'actor_learning_rate':self._actor_learning_rate,
            'critic_learning_rate':self._critic_learning_rate,
            'actor_replay_lambda':self._actor_replay_lambda,
            'critic_replay_lambda':self._critic_replay_lambda,
            'actor_optim_factory':self._actor_optim_factory,
            'critic_optim_factory':self._critic_optim_factory,
            'actor_encoder_factory':self._actor_encoder_factory,
            'critic_encoder_factory':self._critic_encoder_factory,
            'value_encoder_factory':self._value_encoder_factory,
            'gamma':self._gamma,
            'gem_alpha':self._gem_alpha,
            'agem_alpha':self._agem_alpha,
            'ewc_rwalk_alpha':self._ewc_rwalk_alpha,
            'epsilon':self._epsilon,
            "damping":self._damping,
            'tau':self._tau,
            'n_critics':self._n_critics,
            'expectile': self._expectile,
            'weight_temp': self._weight_temp,
            'max_weight': self._max_weight,
            'use_gpu':self._use_gpu,
            'scaler':self._scaler,
            'action_scaler':self._action_scaler,
            'reward_scaler':self._reward_scaler,
            'fine_tuned_step': self._fine_tuned_step,
            "policy_noise": self._policy_noise,
        }
        if self._impl_name in ['iql', 'iql_online']:
            from myd3rlpy.algos.torch.o2o_iql_impl import O2OIQLImpl as O2OImpl
        elif self._impl_name in ['iqln', 'iqln_online']:
            from myd3rlpy.algos.torch.o2o_iqln_impl import O2OIQLNImpl as O2OImpl
        else:
            print(self._impl_name)
            raise NotImplementedError
        self._impl = O2OImpl(
            **impl_dict
        )
        self._impl.build()
