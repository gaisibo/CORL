from myd3rlpy.algos.torch.st_td3_impl import STTD3Impl
from myd3rlpy.algos.torch.st_sac_impl import STSACImpl
from myd3rlpy.algos.torch.st_iql_impl import STIQLImpl
from myd3rlpy.algos.torch.st_cql_impl import STCQLImpl
from myd3rlpy.algos.torch.st_ppo_impl import STPPOImpl
from myd3rlpy.algos.torch.o2o_impl import O2OImpl


class O2OPPOImpl(STPPOImpl, O2OImpl):
    def copy_from_sac(self, sac_impl: STSACImpl, copy_optim: bool):
        self._policy.load_state_dict(sac_impl._policy.state_dict())
        self._targ_policy.load_state_dict(sac_impl._policy.state_dict())
        if copy_optim:
            self._actor_optim.load_state_dict(sac_impl._actor_optim.state_dict())

    def copy_from_cql(self, cql_impl: STCQLImpl, copy_optim: bool):
        self._policy.load_state_dict(cql_impl._policy.state_dict())
        self._build_actor_optim()
        if copy_optim:
            self._actor_optim.load_state_dict(cql_impl._actor_optim.state_dict())

    def copy_from_td3(self, td3_impl: STTD3Impl, copy_optim: bool):
        assert self._policy is not None
        assert self._targ_policy is not None
        assert td3_impl._policy is not None
        assert td3_impl._targ_policy is not None
        policy_state_dict = td3_impl._policy.state_dict()
        policy_state_dict['_mu.weight'] = policy_state_dict['_fc.weight']
        policy_state_dict['_mu.bias'] = policy_state_dict['_fc.bias']
        policy_state_dict['_logstd.weight'] = self._policy._logstd.weight.data
        policy_state_dict['_logstd.bias'] = self._policy._logstd.bias.data
        del policy_state_dict['_fc.weight']
        del policy_state_dict['_fc.bias']
        self._policy.load_state_dict(policy_state_dict)
        self._build_actor_optim()
        if copy_optim:
            td3_actor_optim_state_dict = td3_impl._actor_optim.state_dict()
            actor_optim_state_dict = self._actor_optim.state_dict()
            for i, _ in enumerate(td3_impl._policy.parameters()):
                actor_optim_state_dict['state'][i] = td3_actor_optim_state_dict['state'][i]
            self._actor_optim.load_state_dict(actor_optim_state_dict)

    def copy_from_iql(self, iql_impl: STIQLImpl, copy_optim: bool):
        assert self._policy is not None
        assert self._targ_policy is not None
        assert iql_impl._policy is not None
        assert iql_impl._targ_policy is not None
        self._policy.load_state_dict(iql_impl._policy.state_dict())
        self._build_actor_optim()
        if copy_optim:
            self._actor_optim.load_state_dict(iql_impl._actor_optim.state_dict())
    def copy_from_iqln(self, iql_impl: STIQLImpl, copy_optim: bool):
        self.copy_from_iql(iql_impl, copy_optim)
