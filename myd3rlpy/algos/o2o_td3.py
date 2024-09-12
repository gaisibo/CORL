from typing import Dict, Optional, List
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

    def copy_from_past(self, arg1: str, impl: STImpl, copy_optim: bool):
        if arg1 == 'sac':
            self.copy_from_sac(impl, copy_optim)
        elif arg1 == 'iql':
            self.copy_from_iql(impl, copy_optim)
        elif arg1 == 'td3':
            pass
        else:
            raise NotImplementedError
