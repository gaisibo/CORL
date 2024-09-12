from typing import Dict, Optional, List
from torch import Tensor

from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR

from myd3rlpy.algos.o2o import O2OBase
from myd3rlpy.algos.st_cql import STCQL

class O2OCQL(O2OBase, STCQL):
    # 注意欧氏距离最近邻被塞到actions后面了。
    def _update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool) -> Dict[int, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(value_batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})
        if self._alpha_learning_rate > 0:
            alpha_loss, alpha = self._impl.update_alpha(value_batch)
            metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

        critic_loss, replay_critic_loss = self._impl.update_critic(value_batch, None, clone_critic=self._clone_critic, online=online)
        metrics.update({"critic_loss": critic_loss})
        metrics.update({"replay_critic_loss": replay_critic_loss})

        actor_loss, replay_actor_loss = self._impl.update_actor(policy_batch, None, clone_actor=self._clone_actor, online=online)
        metrics.update({"actor_loss": actor_loss})
        metrics.update({"replay_actor_loss": replay_actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics
