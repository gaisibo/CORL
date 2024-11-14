from myd3rlpy.models.builders import create_parallel_continuous_q_function
from myd3rlpy.algos.torch.o2o_iql_impl import O2OIQLImpl
from myd3rlpy.algos.torch.st_iqln_impl import STIQLNImpl
from utils.networks import ParallelizedEnsembleFlattenMLP


class O2OIQLNImpl(STIQLNImpl, O2OIQLImpl):

    def _build_actor(self):
        super(STIQLNImpl, self)._build_actor()

    def _build_critic(self) -> None:
        # self._value_func = nn.ModuleList([create_value_function(self._observation_shape, self._value_encoder_factory) for _ in range(self._n_ensemble)])
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            reduction='min',
        )
        self._value_func = ParallelizedEnsembleFlattenMLP(self._n_ensemble, [256, 256], self._observation_shape[0], 1, device=self.device)
    #def _build_actor(self) -> None:
    #    self._policy = create_squashed_normal_policy(
    #        self._observation_shape,
    #        self._action_size,
    #        self._actor_encoder_factory,
    #        min_logstd = -6,
    #        max_logstd = 0,
    #        use_std_parameter = True,
    #    )
    #def copy_from_iql(self, iqln_impl: STIQLImpl, copy_optim: bool):
    #    super().copy_from_iql(iqln_impl, copy_optim)
    #def copy_from_iql(self, iqln_impl: STIQLImpl, copy_optim: bool):
    #    raise NotImplementedError, "IQLN cannot load IQL"
