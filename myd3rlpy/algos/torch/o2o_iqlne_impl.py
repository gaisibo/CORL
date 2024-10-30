import torch

from d3rlpy.models.builders import create_squashed_normal_policy

from myd3rlpy.algos.torch.o2o_iql_impl import O2OIQLImpl
from myd3rlpy.algos.torch.st_iqln_impl import STIQLNImpl
from d3rlpy.torch_utility import TorchMiniBatch


class O2OIQLNEImpl(STIQLNImpl, O2OIQLImpl):
    def __init__(self, random_choice_num, **kwargs):
        super().__init__(**kwargs)
        self._random_choice_num = random_choice_num

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            #v_t, _ = torch.min(self._value_func(batch.next_observations), dim=0)
            #v_t = self._value_func(batch.next_observations)
            #sample = torch.randperm(v_t.shape[0], dtype=torch.int32, device="cuda")[:self._random_choice_num]
            #v_t = torch.mean(v_t[sample], dim=0)
            #return v_t
            v_t = self._value_func(batch.next_observations)
            v_t = torch.mean(v_t, dim=0)
            v_t_std = torch.std(v_t, dim=0)
            if self._entropy_time != 0:
                entropy = 0.5 * torch.log(2 * math.pi * math.e * v_t_std)
                v_t += torch.mean(entropy) / self._entropy_time
            return v_t

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        error = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )
        # self._str += str((self._q_func(batch.observations, batch.actions) + batch.rewards - q_tpn * self._gamma).mean()) + ' ' + str(error.mean()) + '\n'
        return error

    def _compute_weight(self, observations, actions) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(observations, actions, "min")
        # v_t = []
        # for value_func in self._value_func:
        #     v_t.append(value_func(observations))
        # v_t, _ = torch.min(torch.stack(v_t, dim=0), dim=0)
        v_t, _ = torch.min(self._value_func(observations), dim=0)
        adv = q_t - v_t
        weight = (self._weight_temp * adv).exp().clamp(max=self._max_weight)

        return weight
