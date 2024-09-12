from typing import Dict, Optional, List
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

        actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, clone_actor=self._clone_actor, online=online)
        # actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
        metrics.update({"actor_loss": actor_loss})
        metrics.update({"replay_actor_loss": replay_actor_loss})

        self._impl.update_critic_target()

        return metrics

    def copy_from_past(self, arg1: str, impl: STImpl, copy_optim: bool):
        if arg1 == 'sac':
            self._impl.copy_from_sac(impl, copy_optim)
        elif arg1 == 'td3':
            self._impl.copy_from_td3(impl, copy_optim)
        elif arg1 == 'iql':
            pass
        else:
            raise NotImplementedError
