import torch
from myd3rlpy.algos.torch.plug.gem import overwrite_grad, store_grad
from myd3rlpy.algos.torch.plug.plug import Plug


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class AGEM(Plug):
    def build(self):
        self.grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in self._networks]
        self.grad_xy = [torch.zeros(np.sum(critic_grad_dims)).to(self.device) for self.grad_dims in self.grad_dims]
        self.grad_er = [torch.zeros(np.sum(critic_grad_dims)).to(self.device) for self.grad_dims in self.grad_dims]

    def pre_loss(self, batch):
        self._update(batch)
        for network, grad_er, grad_dim in zip(self._networks, self.grad_ers, self.grad_dims):
            store_grad(network.parameters(), grad_er, grad_dim)

    def post_loss(self):
        for network, grad_er, grad_dim, grad_xy in zip(self._networks, self.grad_ers, self.grad_dims, self.grad_xys):
            store_grad(network.parameters(), grad_xy, grad_dim)
            dot_prod = torch.dot(grad_xy, grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=grad_xy, ger=grad_er)
                overwrite_grad(network.parameters, g_tilde, grad_dim)
            else:
                overwrite_grad(network.parameters, grad_xy, grad_dim)
