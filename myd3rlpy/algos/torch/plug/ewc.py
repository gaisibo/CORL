import torch
from myd3rlpy.algos.torch.plug.plug import Plug


class EWC(Plug):
    def build(self, networks):
        # Store current parameters for the next task
        self.older_params = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in networks]
        # Store fisher information weight importance
        self.fisher = [{n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters() if p.requires_grad} for network in networks]

    def _add_ewc_loss(self, networks):
        replay_ewc_loss = 0
        for network, fisher, older_param in zip(networks, self.fishers, self.older_params):
            for n, p in network.named_parameters():
                if n in fisher.keys():
                    replay_ewc_loss += torch.mean(fisher[n] * (p - older_param[n]).pow(2)) / 2
        return replay_ewc_loss

    def _ewc_rwalk_post_train_process(self, networks, iterator, optim, update, batch_size=None, n_frames=None, n_steps=None, gamma=None, test=False):
        for i, (network, fisher, older_param) in enumerate(zip(networks, self.fishers, self.older_params)):
            curr_fisher = self.compute_fisher_matrix_diag(iterator, network, optim, update, batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
                fisher[n] = (self._ewc_rwalk_alpha * fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])
