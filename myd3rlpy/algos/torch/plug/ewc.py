import torch
from myd3rlpy.algos.torch.plug.plug import Plug
from myd3rlpy.iterators.base import TransitionIterator


class EWC(Plug):
    def __init__(self, algo, networks):
        super().__init__(algo, networks)
        self._ewc_rwalk_alpha = algo._ewc_rwalk_alpha
    def build(self):
        # Store current parameters for the next task
        self.older_params = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        # Store fisher information weight importance
        self.fishers = [{n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]

    def _add_ewc_loss(self):
        replay_ewc_loss = 0
        for network, fisher, older_param in zip(self._networks, self.fishers, self.older_params):
            for n, p in network.named_parameters():
                if n in fisher.keys():
                    replay_ewc_loss += torch.mean(fisher[n] * (p - older_param[n]).pow(2)) / 2
        return replay_ewc_loss

    def compute_fisher_matrix_diag(self, iterator, network, optim, update, batch_size=None, n_frames=None, n_steps=None, gamma=None, test=False):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        network.train()
        replay_loss = 0
        if isinstance(iterator, TransitionIterator):
            iterator.reset()
        else:
            pass
        for t in range(len(iterator) if not test else 2):
            if isinstance(iterator, TransitionIterator):
                batch = next(iterator)
            else:
                batch = iterator.sample(batch_size=batch_size,
                        n_frames=n_frames,
                        n_steps=n_steps,
                        gamma=gamma)
            optim.zero_grad()
            update(self, batch)
            # Accumulate all gradients from loss with regularization
            for n, p in network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        # Apply mean across all samples
        fisher = {n: (p / len(iterator)) for n, p in fisher.items()}
        return fisher

    def _ewc_rwalk_post_train_process(self, iterator, optim, update, batch_size=None, n_frames=None, n_steps=None, gamma=None, test=False):
        for i, (network, fisher, older_param) in enumerate(zip(self._networks, self.fishers, self.older_params)):
            curr_fisher = self.compute_fisher_matrix_diag(iterator, network, optim, update, batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
                fisher[n] = (self._ewc_rwalk_alpha * fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])
