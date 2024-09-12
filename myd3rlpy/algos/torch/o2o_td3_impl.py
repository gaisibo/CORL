from myd3rlpy.algos.torch.st_td3_impl import STTD3Impl
from myd3rlpy.algos.torch.st_sac_impl import STSACImpl
from myd3rlpy.algos.torch.st_iql_impl import STIQLImpl
from myd3rlpy.algos.torch.st_cql_impl import STCQLImpl


class O2OTD3Imple(STTD3Impl):
    # sac actor: _encoder, _mu.weight, _mu.bias, _logstd.weight, _logstd.bias
    # td3 actor: _encoder, _fc.weight, _fc.bias
    # iql actor: _logstd, _encoder, _fc.weight, _fc.bias
    # sac critic: _encoder
    # td3 critic: _encoder
    # iql critic: q._encoder, value._encoder
    def copy_from_cql(self, cql_impl: STCQLImpl, copy_optim: bool):
        self.copy_from_sac(cql_impl, copy_optim=copy_optim)

    def copy_from_sac(self, sac_impl: STSACImpl, copy_optim: bool):
        assert self._policy is not None
        assert self._targ_policy is not None
        assert sac_impl._policy is not None
        assert sac_impl._targ_policy is not None
        policy_state_dict = sac_impl._policy.state_dict()
        policy_state_dict['_fc.weight'] = policy_state_dict['_mu.weight']
        policy_state_dict['_fc.bias'] = policy_state_dict['_mu.bias']
        del policy_state_dict['_mu.weight']
        del policy_state_dict['_mu.bias']
        del policy_state_dict['_logstd.weight']
        del policy_state_dict['_logstd.bias']
        targ_policy_state_dict = sac_impl._targ_policy.state_dict()
        targ_policy_state_dict['_fc.weight'] = targ_policy_state_dict['_mu.weight']
        targ_policy_state_dict['_fc.bias'] = targ_policy_state_dict['_mu.bias']
        del targ_policy_state_dict['_mu.weight']
        del targ_policy_state_dict['_mu.bias']
        del targ_policy_state_dict['_logstd.weight']
        del targ_policy_state_dict['_logstd.bias']
        #del targ_policy_state_dict['_logstd']
        self._q_func.load_state_dict(sac_impl._q_func.state_dict())
        self._policy.load_state_dict(policy_state_dict)
        self._targ_q_func.load_state_dict(sac_impl._targ_q_func.state_dict())
        self._targ_policy.load_state_dict(targ_policy_state_dict)
        self._build_critic_optim()
        self._build_actor_optim()
        if copy_optim:
            actor_optim_state_dict = sac_impl._actor_optim.state_dict()
            del actor_optim_state_dict['state'][6]
            del actor_optim_state_dict['state'][7]
            actor_optim_state_dict['param_groups'][0]['params'] = list(range(6))
            self._actor_optim.load_state_dict(actor_optim_state_dict)
            self._critic_optim.load_state_dict(sac_impl._critic_optim.state_dict())

    def copy_from_iql(self, iql_impl: STIQLImpl, copy_optim: bool):
        assert self._policy is not None
        assert self._targ_policy is not None
        assert iql_impl._policy is not None
        assert iql_impl._targ_policy is not None
        policy_state_dict = iql_impl._policy.state_dict()
        policy_state_dict['_fc.weight'] = policy_state_dict['_mu.weight']
        policy_state_dict['_fc.bias'] = policy_state_dict['_mu.bias']
        del policy_state_dict['_mu.weight']
        del policy_state_dict['_mu.bias']
        del policy_state_dict['_logstd.weight']
        del policy_state_dict['_logstd.bias']
        targ_policy_state_dict = iql_impl._targ_policy.state_dict()
        targ_policy_state_dict['_fc.weight'] = targ_policy_state_dict['_mu.weight']
        targ_policy_state_dict['_fc.bias'] = targ_policy_state_dict['_mu.bias']
        del targ_policy_state_dict['_mu.weight']
        del targ_policy_state_dict['_mu.bias']
        del targ_policy_state_dict['_logstd.weight']
        del targ_policy_state_dict['_logstd.bias']

        self._q_func.load_state_dict(iql_impl._q_func.state_dict())
        self._policy.load_state_dict(policy_state_dict)
        self._targ_q_func.load_state_dict(iql_impl._targ_q_func.state_dict())
        self._targ_policy.load_state_dict(targ_policy_state_dict)
        self._build_critic_optim()
        self._build_actor_optim()
        if copy_optim:
            actor_optim_state_dict = iql_impl._actor_optim.state_dict()
            del actor_optim_state_dict['state'][6]
            del actor_optim_state_dict['state'][7]
            actor_optim_state_dict['param_groups'][0]['params'] = list(range(6))
            self._actor_optim.load_state_dict(actor_optim_state_dict)

            critic_optim_state_dict = iql_impl._critic_optim.state_dict()
            for i in range(6, 12):
                del critic_optim_state_dict['state'][i]
            critic_optim_state_dict['param_groups'][0]['params'] = list(range(6))
            self._critic_optim.load_state_dict(critic_optim_state_dict)
