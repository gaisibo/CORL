from abc import ABC, abstractmethod


class O2OImpl(ABC):
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
