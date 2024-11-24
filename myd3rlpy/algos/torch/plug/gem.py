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
    def __init__(self, algo, networks):
        super().__init__(algo, networks)
        self._gem_alpha = algo._gem_alpha

    def build(self):
        # Allocate temporary synaptic memory
        self.grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in self._networks]
        #self.grads_cs = [torch.zeros(np.sum(grad_dims)).to(self._networks[0].device) for grad_dims in self.grad_dims]
        self.grads_cs = [dict() for _ in self.grad_dims]
        self.grads_da = [torch.zeros(np.sum(grad_dims)).to(self._networks[0].device) for grad_dims in self.grad_dims]
        #self.grads_cs = [{task_id: torch.zeros(np.sum(grad_dims)).to(self._networks[0].device) for task_id in algo.learned_id} for grad_dims in self.grad_dims]

    def pre_loss(self, batch):
        self._update(batch)
        for network_id, (network, grads_cs, grad_dim) in enumerate(zip(self._networks, self.grads_cs, self.grad_dims)):
            if self._algo._impl_id not in grads_cs.keys():
                self.grads_cs[network_id][self._algo._impl_id] = torch.zeros(np.sum(grad_dim)).to(self._networks[0].device)
            store_grad(network.parameters(), grads_cs[self._algo._impl_id], grad_dim)

    def post_loss(self):
        for network, grads_da, grad_dim, grads_cs in zip(self._networks, self.grads_da, self.grad_dims, self.grads_cs):
            # copy gradient
            store_grad(network.parameters(), grads_da, grad_dim)
            grads_cs = torch.stack([grads_cs[task_id] for task_id in grads_cs.keys()], dim=0)
            dot_prod = torch.dot(grads_da, grads_cs)
            if (dot_prod < 0).sum() != 0:
                # project2cone2(self._actor_grads_da.unsqueeze(1),
                #               torch.stack(list(self._actor_grads_cs).values()).T, margin=self._gem_alpha)
                project2cone2(grads_da.unsqueeze(dim=1), grads_cs, margin=self._gem_alpha)
                # copy gradients back
                overwrite_grad(network.parameters, grads_da, grad_dim)
