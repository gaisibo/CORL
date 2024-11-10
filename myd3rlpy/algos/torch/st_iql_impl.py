import torch

from d3rlpy.models.builders import create_non_squashed_normal_policy, create_value_function
from d3rlpy.torch_utility import TorchMiniBatch, eval_api
from d3rlpy.algos.torch.iql_impl import IQLImpl

from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.models.builders import create_parallel_continuous_q_function
from myd3rlpy.torch_utility import torch_api
from utils.networks import ParallelizedEnsembleFlattenMLP


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STIQLImpl(STImpl, IQLImpl):

    def __init__(
        self,
        policy_noise,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self._policy_noise = policy_noise

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            reduction='min',
        )
        self._value_func = ParallelizedEnsembleFlattenMLP(self._n_critics, [256, 256], self._observation_shape[0], 1, device=self.device)
        self._critic_networks = [self._q_func, self._value_func]

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool=False, online: bool=False, replay: bool = False) -> torch.Tensor:
        assert self._policy

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch.observations, batch.actions)
        ret = (-weight * log_probs).mean()
        # if not replay:
        #     self._log_probs = log_probs.mean()
        # else:
        #     self._replay_log_probs = log_probs.mean()
        #if clone_actor and not online and self._clone_policy is not None:
        #    # compute log probability
        #    dist = self._clone_policy.dist(batch.observations)
        #    log_probs = dist.log_prob(self._clone_policy(batch.observations))

        #    # compute weight
        #    with torch.no_grad():
        #        weight = self._compute_weight(batch.observations, self._clone_policy(batch.observations))
        #    ret += -(weight * log_probs).mean()

        return ret

    def _compute_weight(self, observations, actions) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(observations, actions, "min")
        v_t = self._value_func(observations)
        adv = q_t - v_t
        weight = (self._weight_temp * adv).exp().clamp(max=self._max_weight).detach()
        return weight

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            v_t = self._value_func(batch.next_observations)
            return v_t

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def _compute_value_loss(self, batch: TorchMiniBatch, clone_critic=False, replay: bool = False) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        # if clone_critic and '_clone_value_func' in self.__dict__.keys() and '_clone_q_func' in self.__dict__.keys():
        #     clone_v_t = self._clone_value_func(batch.observations).detach()
        #     diff_clone = (clone_v_t - v_t)
        #     diff = torch.max(diff, diff_clone)
        # else:
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        ret = (weight * (diff ** 2)).mean()
        return ret

    def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=True, online: bool = False, replay=False, first_time=False):
        assert self._q_func is not None
        critic_loss = self._compute_critic_loss(batch, q_tpn)
        value_loss = self._compute_value_loss(batch, clone_critic=clone_critic, replay=replay)
        self._q_loss = critic_loss.mean()
        self._v_loss = value_loss.mean()
        return critic_loss + value_loss

    def compute_generate_critic_loss(self, batch, clone_critic: bool=True):
        assert self._q_func is not None
        return self._compute_value_loss(batch, clone_critic=clone_critic)

    def compute_actor_loss(self, batch, clone_actor: bool = False, online: bool = False, replay: bool = False):
        return self._compute_actor_loss(batch, clone_actor=clone_actor, online=online, replay=replay)

    #@eval_api
    #@torch_api(scaler_targets=["x"])
    #def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
    #    action = super()._predict_best_action(x)
    #    noise = torch.randn_like(action) * self._policy_noise
    #    action = torch.clamp(action + noise, -1.0, 1.0)
    #    return action#.cpu().detach().numpy()
