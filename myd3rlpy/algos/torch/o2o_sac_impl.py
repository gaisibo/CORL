from typing import Tuple
import torch
import numpy as np
from myd3rlpy.algos.torch.st_td3_impl import STTD3Impl
from myd3rlpy.algos.torch.st_sac_impl import STSACImpl
from myd3rlpy.algos.torch.st_iql_impl import STIQLImpl
from myd3rlpy.algos.torch.st_cql_impl import STCQLImpl
from myd3rlpy.algos.torch.o2o_impl import O2OImpl
from myd3rlpy.torch_utility import torch_api, TorchMiniBatch
from d3rlpy.torch_utility import train_api


class O2OSACImpl(STSACImpl, O2OImpl):
    # sac actor: _encoder, _mu.weight, _mu.bias, _logstd.weight, _logstd.bias
    # td3 actor: _encoder, _fc.weight, _fc.bias
    # iql actor: _logstd, _encoder, _fc.weight, _fc.bias
    # sac critic: _encoder
    # td3 critic: _encoder
    # iql critic: q._encoder, value._encoder

    def copy_from_sac(self, sac_impl: STSACImpl, copy_optim: bool):
        self._q_func.load_state_dict(sac_impl._q_func.state_dict())
        self._policy.load_state_dict(sac_impl._policy.state_dict())
        self._targ_q_func.load_state_dict(sac_impl._targ_q_func.state_dict())
        self._targ_policy.load_state_dict(sac_impl._policy.state_dict())
        self._log_temp.load_state_dict(sac_impl._log_temp.state_dict())
        if copy_optim:
            self._actor_optim.load_state_dict(sac_impl._actor_optim.state_dict())
            self._critic_optim.load_state_dict(sac_impl._critic_optim.state_dict())
            self._temp_optim.load_state_dict(sac_impl._temp_optim.state_dict())

    def copy_from_cql(self, cql_impl: STCQLImpl, copy_optim: bool):
        self._q_func.load_state_dict(cql_impl._q_func.state_dict())
        self._policy.load_state_dict(cql_impl._policy.state_dict())
        self._targ_q_func.load_state_dict(cql_impl._targ_q_func.state_dict())
        self._targ_policy.load_state_dict(cql_impl._policy.state_dict())
        self._log_temp.load_state_dict(cql_impl._log_temp.state_dict())
        if copy_optim:
            self._actor_optim.load_state_dict(cql_impl._actor_optim.state_dict())
            self._critic_optim.load_state_dict(cql_impl._critic_optim.state_dict())
            self._temp_optim.load_state_dict(cql_impl._temp_optim.state_dict())

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
        targ_policy_state_dict = td3_impl._targ_policy.state_dict()
        targ_policy_state_dict['_mu.weight'] = targ_policy_state_dict['_fc.weight']
        targ_policy_state_dict['_mu.bias'] = targ_policy_state_dict['_fc.bias']
        targ_policy_state_dict['_logstd.weight'] = self._targ_policy._logstd.weight.data
        targ_policy_state_dict['_logstd.bias'] = self._targ_policy._logstd.bias.data
        del targ_policy_state_dict['_fc.weight']
        del targ_policy_state_dict['_fc.bias']

        self._q_func.load_state_dict(td3_impl._q_func.state_dict())
        self._policy.load_state_dict(policy_state_dict)
        self._targ_q_func.load_state_dict(td3_impl._targ_q_func.state_dict())
        self._targ_policy.load_state_dict(targ_policy_state_dict)
        self._build_critic_optim()
        self._build_actor_optim()
        if copy_optim:
            td3_actor_optim_state_dict = td3_impl._actor_optim.state_dict()
            actor_optim_state_dict = self._actor_optim.state_dict()
            for i, _ in enumerate(td3_impl._policy.parameters()):
                actor_optim_state_dict['state'][i] = td3_actor_optim_state_dict['state'][i]
            self._actor_optim.load_state_dict(actor_optim_state_dict)
            self._critic_optim.load_state_dict(td3_impl._critic_optim.state_dict())

    def copy_from_iql(self, iql_impl: STIQLImpl, copy_optim: bool):
        assert self._policy is not None
        assert self._targ_policy is not None
        assert iql_impl._policy is not None
        assert iql_impl._targ_policy is not None
        self._q_func.load_state_dict(iql_impl._q_func.state_dict())
        self._policy.load_state_dict(iql_impl._policy.state_dict())
        self._targ_q_func.load_state_dict(iql_impl._targ_q_func.state_dict())
        # iql do not have a targ_policy
        self._targ_policy.load_state_dict(iql_impl._policy.state_dict())
        self._build_critic_optim()
        self._build_actor_optim()
        if copy_optim:
            self._actor_optim.load_state_dict(iql_impl._actor_optim.state_dict())
            iql_critic_optim_state_dict = iql_impl._critic_optim.state_dict()
            critic_optim_state_dict = self._critic_optim.state_dict()
            for i, _ in enumerate(self._q_func.parameters()):
                critic_optim_state_dict['state'][i] = iql_critic_optim_state_dict['state'][i]
            self._critic_optim.load_state_dict(critic_optim_state_dict)
    def copy_from_iqln(self, iql_impl: STIQLImpl, copy_optim: bool):
        self.copy_from_iql(iql_impl, copy_optim)

    @train_api
    @torch_api()
    def update_temp(
        self, batch: TorchMiniBatch
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None

        self._temp_optim.zero_grad()

        with torch.no_grad():
            _, log_prob = self._policy.sample_with_log_prob(batch.observations)
            targ_temp = log_prob - self._action_size

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp
