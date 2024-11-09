from d3rlpy.models.builders import create_squashed_normal_policy

from myd3rlpy.algos.torch.o2o_iql_impl import O2OIQLImpl
from myd3rlpy.algos.torch.st_iqln_impl import STIQLNImpl


class O2OIQLNImpl(STIQLNImpl, O2OIQLImpl):
    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd = -6,
            max_logstd = 0,
            use_std_parameter = True,
        )
    #def copy_from_iql(self, iqln_impl: STIQLImpl, copy_optim: bool):
    #    super().copy_from_iql(iqln_impl, copy_optim)
    #def copy_from_iql(self, iqln_impl: STIQLImpl, copy_optim: bool):
    #    raise NotImplementedError, "IQLN cannot load IQL"
