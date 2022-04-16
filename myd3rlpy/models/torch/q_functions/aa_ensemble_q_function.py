import torch
from torch import nn
from d3rlpy.models.torch.q_functions import DiscreteQFunction, ContinuousQFunction
from d3rlpy.models.torch.q_functions.ensemble_q_function import EnsembleQFunction


class AAEnsembleQFunction(EnsembleQFunction):
    def __init__(self, q_funcs: Union[List[DiscreteQFunction], List[ContinuousQFunction]]):
        super().__init__(q_funcs)
        self.alphas = nn.ParameterList([nn.Parameter(nn.full((1,), 1 / len(q_funcs))) for _ in range(q_funcs)])

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float = 0.99)
    assert target.ndim == 2

    td_sum = torch.tensor(0.0, dtype=torch.float32, device=observations.device)
    for q_func in self._q_funcs:
        if 
        loss = q_func.compute_error(
                observations=observations,
                actions=actions,
                rewards=rewards,
                target=target,
                terminals=terminals,
                gamma=gamma,
                reduction="none",
                )
