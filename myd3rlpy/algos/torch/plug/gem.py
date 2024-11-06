import numpy as np
import torch
import quadprog
from myd3rlpy.algos.torch.plug.plug import Plug


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params:
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))

class GEM(Plug):
    def build(self, networks):
        # Allocate temporary synaptic memory
        self.grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in networks]
        self.grads_cs = [torch.zeros(np.sum(grad_dims)).to(self.device) for grad_dims in self.grad_dims]
        self.grads_da = [torch.zeros(np.sum(grad_dims)).to(self.device) for grad_dims in self.grad_dims]

    def _pre_gem_loss(self, networks):
        for network, grads_cs, grad_dim in zip(networks, self.grads_cs, self.grad_dims):
            store_grad(network.parameters(), grads_cs, grad_dim)

    def _pos_gem_loss(self, networks):
        for network, grads_da, grad_dim, grads_cs in zip(networks, self.grads_das, self.grad_dims, self.grads_css):
            # copy gradient
            store_grad(network.parameters(), grads_da, grad_dim)
            dot_prod = torch.dot(grads_da, grads_cs)
            if (dot_prod < 0).sum() != 0:
                # project2cone2(self._actor_grads_da.unsqueeze(1),
                #               torch.stack(list(self._actor_grads_cs).values()).T, margin=self._gem_alpha)
                project2cone2(grads_da.unsqueeze(dim=1), grads_cs.unsqueeze(dim=1), margin=self._gem_alpha)
                # copy gradients back
                overwrite_grad(network.parameters, grads_da, grad_dim)
