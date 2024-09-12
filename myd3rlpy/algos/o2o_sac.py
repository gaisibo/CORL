from typing import Dict, Optional, List
from torch import Tensor

from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR

from myd3rlpy.algos.o2o import O2OBase
from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.algos.st_sac import STSAC

class O2OSAC(O2OBase, STSAC):
    # 注意欧氏距离最近邻被塞到actions后面了。
    def _update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(value_batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        critic_loss, replay_critic_loss = self._impl.update_critic(value_batch, None, clone_critic=self._clone_critic, online=online)
        metrics.update({"critic_loss": critic_loss})
        metrics.update({"replay_critic_loss": replay_critic_loss})

        if self._grad_step % self._update_actor_interval == 0:
            actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, clone_actor=self._clone_actor, online=online)
            # actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
            metrics.update({"actor_loss": actor_loss})
            metrics.update({"replay_actor_loss": replay_actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def copy_from_past(self, arg1: str, impl: STImpl, copy_optim: bool):
        assert self._impl is not None
        if arg1 == 'td3':
            self._impl.copy_from_td3(impl, copy_optim)
        elif arg1 == 'iql':
            self._impl.copy_from_iql(impl, copy_optim)
        elif arg1 == 'sac':
            pass
        else:
            raise NotImplementedError
