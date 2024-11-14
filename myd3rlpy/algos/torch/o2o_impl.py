from d3rlpy.models.builders import create_squashed_normal_policy
from abc import ABC, abstractmethod


class O2OImpl(ABC):
    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd = -6,
            max_logstd = 0,
            use_std_parameter = True,
        )
    @abstractmethod
    def copy_from_sac(self, sac_impl, copy_optim):
        pass
    @abstractmethod
    def copy_from_cql(self, sac_impl, copy_optim):
        pass
    @abstractmethod
    def copy_from_td3(self, sac_impl, copy_optim):
        pass
    @abstractmethod
    def copy_from_iql(self, sac_impl, copy_optim):
        pass
    #@abstractmethod
    #def copy_from_iqln(self, sac_impl, copy_optim):
    #    pass
