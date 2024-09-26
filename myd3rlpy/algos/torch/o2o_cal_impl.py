import torch

from d3rlpy.models.builders import create_non_squashed_normal_policy
from d3rlpy.torch_utility import TorchMiniBatch

from myd3rlpy.algos.torch.o2o_cql_impl import O2OCQLImpl
from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.models.builders import create_parallel_continuous_q_function


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class O2OCALImpl(O2OCQLImpl):

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool = False, online: bool = False, replay: bool=False, first_time = False
    ) -> torch.Tensor:
        assert self._q_func is not None
        loss = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )
        if online:
            return loss.mean()
        # if clone_critic:
        #     action = self._policy(batch.observations)
        #     q_t = self._q_func(batch.observations, action, "min")
        #     clone_action = self._clone_policy.sample(batch.observations)
        #     clone_q_t = self._clone_q_func(batch.observations, clone_action, "min")
        #     conservative_loss = self._compute_conservative_loss(
        #         batch.observations, batch.actions, batch.next_observations
        #     )
        #     loss += conservative_loss
        #     loss = torch.where(q_t > clone_q_t, loss, torch.zeros_like(loss)).mean()
        # else:
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations, batch.rtgs
        )
        loss += conservative_loss.mean()
        return loss

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor, rtg_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        random_values = self._compute_random_is_values(obs_t)

        lower_bounds = rtg_t.reshape(1, -1, 1).repeat(2, 1, policy_values_t.shape[-1])
        assert lower_bounds.shape == policy_values_t.shape
        num_vals = torch.numel(policy_values_t)
        #bound_rate_policy_values_t = torch.sum(policy_values_t < lower_bounds, dim=0)
        #bound_rate_policy_values_tp1 = torch.sum(policy_values_tp1 < lower_bounds, dim=0)
        policy_values_t = torch.max(policy_values_t, lower_bounds)
        policy_values_tp1 = torch.max(policy_values_tp1, lower_bounds)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t, "none")

        loss = logsumexp.mean(dim=0).mean() - data_values.mean(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)
