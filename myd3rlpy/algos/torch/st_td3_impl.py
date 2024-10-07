import torch

from d3rlpy.torch_utility import TorchMiniBatch, eval_api
from d3rlpy.algos.torch.td3_impl import TD3Impl

from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.models.builders import create_parallel_continuous_q_function
from myd3rlpy.torch_utility import torch_api


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STTD3Impl(STImpl, TD3Impl):

    def __init__(self, policy_noise, **kwargs):
        self._policy_noise = policy_noise
        super().__init__(**kwargs)

    def _build_critic(self) -> None:
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            reduction='min',
        )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool = False, online: bool = False, first_time: bool = False
    ) -> torch.Tensor:
        return super().compute_critic_loss(batch, q_tpn)

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool = False, online: bool = False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action)[0]
        if clone_actor and not online:
            clone_q_t = self._q_func(batch.observations, action)[0]
            return - torch.where(q_t > clone_q_t, q_t, torch.zeros_like(q_t)).mean()
        return -q_t.mean()

    def compute_actor_loss(self, batch, clone_actor: bool = False, online: bool = False, replay: bool = False):
        loss = self._compute_actor_loss(batch, clone_actor=clone_actor, online=online)
        return loss

    @eval_api
    @torch_api(scaler_targets=["x"])
    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        action = super()._sample_action(x)
        noise = torch.randn_like(action) * self._policy_noise
        action = torch.clamp(action + noise, -1.0, 1.0)
        return action.cpu().detach().numpy()
