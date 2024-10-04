import torch

from d3rlpy.torch_utility import TorchMiniBatch

from myd3rlpy.algos.torch.co_sac_impl import COSACImpl
#from myd3rlpy.models.torch.q_functions.ensemble_q_function import ParallelEnsembleContinuousQFunction
from myd3rlpy.models.builders import create_parallel_continuous_q_function


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class COSACNImpl(COSACImpl):

    def _build_critic(self) -> None:
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            # 根据Why So Pessimistic?，这里应该是每个Q单独计算自己的target。
            reduction='min',
        )