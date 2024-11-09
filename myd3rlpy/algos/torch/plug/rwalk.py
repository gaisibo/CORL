import torch
from myd3rlpy.algos.torch.plug.plug import Plug
from myd3rlpy.algos.torch.plug.ewc import EWC


class RWalk(EWC):
    def __init__(self, algo, networks):
        super().__init__(algo, networks)
        self._damping = algo._damping

    def build(self):
        # Store current parameters for the next task
        older_params = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        # Store fisher information weight importance
        fisher = [{n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        W = [{n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        scores = [{n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        return older_params, fisher

    def _add_ewc_loss(self):
        replay_ewc_loss = 0
        for network, fisher, older_param in zip(self._networks, self.fishers, self.older_params):
            for n, p in network.named_parameters():
                if n in fisher.keys():
                    replay_ewc_loss += torch.mean(fisher[n] * (p - older_param[n]).pow(2)) / 2
        return replay_ewc_loss

    def _pre_rwalk_loss(self):
        unreg_grads = [{n: p.grad.clone().detach() for n, p in network.named_parameters() if p.grad is not None} for network in self._networks]
        replay_rwalk_loss = 0
        for network, fisher, score, older_param in zip(self._networks, self.fishers, self.scores, self.older_params):
            for n, p in network.named_parameters():
                if n in fisher.keys():
                    replay_rwalk_loss += torch.mean((fisher[n] + score[n]) * (p - older_param[n]).pow(2)) / 2
        return unreg_grads, replay_rwalk_loss

    def _pos_rwalk_loss(self, unreg_grads, Ws, curr_feat_exts):
        for network, grad, W, curr_feat_ext in zip(self._networks, unreg_grads, Ws, curr_feat_exts):
            for n, p in network.named_parameters():
                if n in grad.keys():
                    W[n] -= grad[n] * (p.detach() - curr_feat_ext[n])

    def _ewc_rwalk_post_train_process(self, iterator, optim, update, batch_size=None, n_frames=None, n_steps=None, gamma=None, test=False):
        for i, (network, fisher, score, W, older_param) in enumerate(zip(self._networks, self.fishers, self.scores, self.Ws, self.older_params)):
            curr_fisher = self.compute_fisher_matrix_diag(iterator, network, optim, update, batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
                fisher[n] = (self._ewc_rwalk_alpha * fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])

            # Page 7: Optimization Path-based Parameter Importance: importance scores computation
            curr_critic_score = {n: torch.zeros(p.shape).to(p.device) for n, p in network.named_parameters() if p.requires_grad}
            with torch.no_grad():
                curr_critic_params = {n: p for n, p in network.named_parameters() if p.requires_grad}
                for n, p in score.items():
                    curr_critic_score[n] = W[n] / (
                            fisher[n] * ((curr_critic_params[n] - older_param[n]) ** 2) + self._damping)
                    W[n].zero_()
                    # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                    curr_critic_score[n] = torch.nn.functional.relu(curr_critic_score[n])
                    older_param[n].data = curr_critic_params[n].data.clone()
            # Page 8: alleviating regularization getting increasingly rigid by averaging scores
            for n, p in score.items():
                score[n] = (p + curr_critic_score[n]) / 2
