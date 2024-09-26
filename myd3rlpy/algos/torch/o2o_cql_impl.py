import numpy as np
from d3rlpy.models.builders import create_squashed_normal_policy

from myd3rlpy.algos.torch.st_td3_impl import STTD3Impl
from myd3rlpy.algos.torch.st_sac_impl import STSACImpl
from myd3rlpy.algos.torch.st_iql_impl import STIQLImpl
from myd3rlpy.algos.torch.st_cql_impl import STCQLImpl
from myd3rlpy.algos.torch.o2o_sac_impl import O2OSACImpl
from myd3rlpy.torch_utility import torch_api, TorchMiniBatch
from d3rlpy.torch_utility import train_api


class O2OCQLImpl(STCQLImpl, O2OSACImpl):
    # sac actor: _encoder, _mu.weight, _mu.bias, _logstd.weight, _logstd.bias
    # td3 actor: _encoder, _fc.weight, _fc.bias
    # iql actor: _logstd, _encoder, _fc.weight, _fc.bias
    # sac critic: _encoder
    # td3 critic: _encoder
    # iql critic: q._encoder, value._encoder
    # According to AWAC, use the same modules as SAC
    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

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
        self._log_alpha.load_state_dict(cql_impl._log_alpha.state_dict())
        if copy_optim:
            self._actor_optim.load_state_dict(cql_impl._actor_optim.state_dict())
            self._critic_optim.load_state_dict(cql_impl._critic_optim.state_dict())
            self._temp_optim.load_state_dict(cql_impl._temp_optim.state_dict())
            self._alpha_optim.load_state_dict(cql_impl._alpha_optim.state_dict())

    #def copy_from_sac(self, sac_impl: STSACImpl, copy_optim: bool):
    #    assert self._policy is not None
    #    assert self._targ_policy is not None
    #    assert sac_impl._policy is not None
    #    assert sac_impl._targ_policy is not None
    #    self._q_func.load_state_dict(sac_impl._q_func.state_dict())
    #    self._policy.load_state_dict(sac_impl._policy.state_dict())
    #    self._targ_q_func.load_state_dict(sac_impl._targ_q_func.state_dict())
    #    self._build_critic_optim()
    #    self._build_actor_optim()
    #    if copy_optim:
    #        self._actor_optim.load_state_dict(self._actor_optim.state_dict())
    #        self._critic_optim.load_state_dict(self._critic_optim.state_dict())

    #def copy_from_td3(self, td3_impl: STTD3Impl, copy_optim: bool):
    #    assert self._policy is not None
    #    assert self._targ_policy is not None
    #    assert td3_impl._policy is not None
    #    assert td3_impl._targ_policy is not None
    #    policy_state_dict = td3_impl._policy.state_dict()
    #    policy_state_dict['_mu.weight'] = policy_state_dict['_fc.weight']
    #    policy_state_dict['_mu.bias'] = policy_state_dict['_fc.bias']
    #    # init
    #    policy_state_dict['_logstd.weight'] = self._policy._logstd.weight.data
    #    policy_state_dict['_logstd.bias'] = self._policy._logstd.bias.data
    #    targ_policy_state_dict = td3_impl._targ_policy.state_dict()
    #    targ_policy_state_dict['_mu.weight'] = targ_policy_state_dict['_fc.weight']
    #    targ_policy_state_dict['_mu.bias'] = targ_policy_state_dict['_fc.bias']
    #    # init
    #    targ_policy_state_dict['_logstd.weight'] = self._targ_policy._logstd.weight.data
    #    targ_policy_state_dict['_logstd.bias'] = self._targ_policy._logstd.bias.data

    #    self._q_func.load_state_dict(td3_impl._q_func.state_dict())
    #    self._policy.load_state_dict(policy_state_dict)
    #    self._targ_q_func.load_state_dict(td3_impl._targ_q_func.state_dict())
    #    self._targ_policy.load_state_dict(targ_policy_state_dict)
    #    self._build_critic_optim()
    #    self._build_actor_optim()
    #    if copy_optim:
    #        td3_actor_optim_state_dict = td3_impl._actor_optim.state_dict()
    #        actor_optim_state_dict = self._actor_optim.state_dict()
    #        for i in range(6):
    #            actor_optim_state_dict['state'][i] = td3_actor_optim_state_dict['state'][i]
    #        self._actor_optim.load_state_dict(actor_optim_state_dict)
    #        self._critic_optim.load_state_dict(td3_impl._critic_optim.state_dict())

    #def copy_from_iql(self, iql_impl: STIQLImpl, copy_optim: bool):
    #    assert self._policy is not None
    #    assert self._targ_policy is not None
    #    assert iql_impl._policy is not None
    #    assert iql_impl._targ_policy is not None
    #    self._q_func.load_state_dict(iql_impl._q_func.state_dict())
    #    self._policy.load_state_dict(iql_impl._policy.state_dict())
    #    self._targ_q_func.load_state_dict(iql_impl._targ_q_func.state_dict())
    #    # iql do not have a targ_policy
    #    self._targ_policy.load_state_dict(iql_impl._policy.state_dict())
    #    self._build_critic_optim()
    #    self._build_actor_optim()
    #    if copy_optim:
    #        self._actor_optim.load_state_dict(iql_impl._actor_optim.state_dict())
    #        critic_optim_state_dict = iql_impl._critic_optim.state_dict()
    #        for i in range(6, 12):
    #            del critic_optim_state_dict['state'][i]
    #        self._critic_optim.load_state_dict(critic_optim_state_dict)

    @train_api
    @torch_api()
    def update_alpha(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._alpha_optim.zero_grad()

        # the original implementation does scale the loss value
        loss = -self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )

        loss.backward()
        self._alpha_optim.step()

        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha
