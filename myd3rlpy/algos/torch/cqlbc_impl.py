import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from d3rlpy.algos.torch.cql_impl import CQLImpl


class CQLBCImpl(CQLImpl):
    @train_api
    @torch_api()
    def replay_update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        loss = self.replay_compute_critic_loss(batch)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def replay_compute_critic_loss(self, batch: List[Tensor]) -> torch.Tensor:
        obs_, act_, _, _, q_ = batch
        q = self.q_function(obs_, act_)
        return F.mse_loss(q, q_)

    @train_api
    @torch_api()
    def replay_update_actor(self, batch: List[Tensor]) -> np.ndarray:
        assert self._actor_optim is not None

        self._actor_optim.zero_grad()

        loss = self.replay_compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def replay_compute_actor_loss(self, batch: List[Tensor]) -> torch.Tensor:
        obs_, act_, mean_, stddev_, _ = batch
        dist_ = Normal(mean_, stddev_)
        dist = self._policy.dist(obs_)
        return kl_divergence(dist_, dist)
