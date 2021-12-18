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
    @eval_api
    def predict_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert x.ndim > 1, "Input must have batch dimension."
        assert x.shape[0] == action.shape[0]
        assert self._q_func is not None

        with torch.no_grad():
            values = self._q_func(x, action, task_id).cpu().detach().numpy()
            values = np.transpose(values, [1, 0, 2])

        mean_values = values.mean(axis=1).reshape(-1)
        stds = np.std(values, axis=1).reshape(-1)

        if with_std:
            return mean_values, stds

        return mean_values
