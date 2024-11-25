import torch
from myd3rlpy.algos.torch.plug.plug import Plug


class SI(Plug):
    def __init__(self, *args):
        super().__init__(*args)
        self._epsilon = algo._epsilon

    def build(self):
        # Store current parameters for the next task
        self.older_params = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        self.W = [{n: p.clone().detach().zero_() for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]
        self.omega = [{n: p.clone().detach().zero_() for n, p in network.named_parameters() if p.requires_grad} for network in self._networks]

    def pre_loss(self):
        for network, older_param, W, omega in zip(self._networks, self.older_params, self.Ws, self.omegas):
            for n, p in network.named_parameters():
                if p.grad is not None and n in W.keys():
                    p_change = p.detach().clone() - older_param[n]
                    W[n].add_(-p.grad * p_change)
                    omega_add = W[n] / (p_change ** 2 + self._epsilon)
                    omega_old = omega[n]
                    omega_new = omega_old + omega_add
                    omega[n] = omega_new
        replay_si_loss = 0
        for network, omega, older_param in zip(self._networks, omegas, older_params):
            for n, p in network.named_parameters():
                if p.requires_grad:
                    replay_si_loss += torch.mean(omega[n] * (p - older_param[n]) ** 2)
                older_param[n].data = p.data.clone()
        return replay_si_loss