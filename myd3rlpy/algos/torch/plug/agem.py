import torch
from myd3rlpy.algos.torch.plug.gem import overwrite_grad, store_grad
from myd3rlpy.algos.torch.plug.plug import Plug


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class AGEM(Plug):
    def build(self):
        self.grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in networks]
        self.grad_xy = [torch.zeros(np.sum(critic_grad_dims)).to(networks[0].device) for self.grad_dims in self.grad_dims]
        self.grad_er = [torch.zeros(np.sum(critic_grad_dims)).to(networks[0].device) for self.grad_dims in self.grad_dims]

    def _pre_agem_loss(self, networks):
        for network, grad_xy, grad_dim in zip(networks, self.grad_xys, self.grad_dims):
            store_grad(network.parameters(), grad_xy, grad_dim)

    def _pos_agem_loss(self, networks):
        for network, grad_er, grad_dim, grad_xy in zip(networks, self.grad_ers, self.grad_dims, self.grad_xys):
            store_grad(network.parameters(), grad_er, grad_dim)
            dot_prod = torch.dot(grad_xy, grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=grad_xy, ger=grad_er)
                overwrite_grad(network.parameters, g_tilde, grad_dim)
            else:
                overwrite_grad(network.parameters, grad_xy, grad_dim)
