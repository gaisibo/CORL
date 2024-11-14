import copy
import torch
import math

from d3rlpy.models.builders import create_squashed_normal_policy
from d3rlpy.torch_utility import TorchMiniBatch, eval_api, soft_sync

from myd3rlpy.algos.torch.o2o_iqln_impl import O2OIQLNImpl
from myd3rlpy.torch_utility import torch_api
from myd3rlpy.models.builders import create_parallel_continuous_q_function
from utils.networks import ParallelizedEnsembleFlattenMLP


class O2OIQLNEImpl(O2OIQLNImpl):
    def __init__(self, random_choice_num, **kwargs):
        super().__init__(**kwargs)
        self._random_choice_num = random_choice_num

    def _build_critic(self):
        super()._build_critic()
        self._targ_value_func = copy.deepcopy(self._value_func)

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)
        soft_sync(self._targ_value_func, self._value_func, self._tau)

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool=False, online: bool=False, replay:bool = False) -> torch.Tensor:
        assert self._policy

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch.observations, batch.actions)
        ret = -(weight * log_probs).mean()
        self._iql_loss = ret

        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "mean")[0]
        self._td3_loss = (weight * q_t).mean()
        #ret -= self._td3_loss
        ret = - self._td3_loss
        #return -self._td3_loss
        #return self._iql_loss
        return ret

    def _compute_weight(self, observations, actions) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(observations, actions, "min")
        v_t, _ = torch.min(self._value_func(observations), dim=0)
        adv = q_t - v_t
        self._adv = torch.mean(adv)
        weight = (self._weight_temp * adv).exp().clamp(max=self._max_weight)
        self._weight = torch.mean(weight)

        return weight

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = 0.2 * noise
            clipped_noise = scaled_noise.clamp(
                -0.5, 0.5
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            targ_q = self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )
            v_t, _ = torch.min(self._value_func(batch.next_observations), dim=0)
            #return torch.max(targ_q, v_t)
            return targ_q

    def _compute_value_loss(self, batch: TorchMiniBatch, clone_critic=False, replay=False, first_time=False) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        targ_v_ts = batch.rewards + self._gamma * self._targ_value_func(batch.next_observations)
        v_ts = self._value_func(batch.observations)
        diff = targ_v_ts - v_ts
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        ret = torch.mean(torch.sum(weight * (diff ** 2), dim=0))
        return ret

    @eval_api
    @torch_api(scaler_targets=["x"])
    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        action = super()._predict_best_action(x)
        noise = torch.randn_like(action) * 0.2
        action = torch.clamp(action + noise, -1.0, 1.0)
        return action#.cpu().detach().numpy()
