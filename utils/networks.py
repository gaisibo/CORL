import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def identity(x):
    return x
def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)




class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            device,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.device = device

        self.fcs = []
        self.batch_norms = []

        # data normalization
        self.input_mu = nn.Parameter(torch.zeros(input_size).to(self.device), requires_grad=False).float()
        self.input_std = nn.Parameter(torch.ones(input_size).to(self.device), requires_grad=False).float()

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            hidden_init(fc.weight, w_scale)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.batch_norm:
                bn = nn.BatchNorm1d(next_size)
                self.__setattr__('batch_norm%d' % i, bn)
                self.batch_norms.append(bn)

            in_size = next_size

        self.last_fc = nn.Linear(in_size, output_size)
        if final_init_scale is None:
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            torch.nn.init.orthogonal_(self.last_fc.weight, final_init_scale)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = (input - self.input_mu) / (self.input_std + 1e-6)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std != std] = 0
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std = mask * std + (1-mask) * np.ones(self.input_size)
        self.input_mu.data = torch.from_numpy(mean).to(self.device)
        self.input_std.data = torch.from_numpy(std).to(self.device)


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class ParallelizedLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        device,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        self.device = device

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim)).to(self.device)
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim), dtype=torch.float32).to(self.device)
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsemble(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            device,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.0,
            layer_norm=False,
            layer_norm_kwargs=None,
            spectral_norm=False,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.device = device

        # data normalization
        self.input_mu = nn.Parameter(
            torch.zeros(input_size).to(self.device), requires_grad=False).float()
        self.input_std = nn.Parameter(
            torch.ones(input_size).to(self.device), requires_grad=False).float()

        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            layer_size = (ensemble_size, in_size, next_size)
            fc = ParallelizedLayer(
                ensemble_size, in_size, next_size,
                w_std_value=1/(2*np.sqrt(in_size)),
                b_init_value=b_init_value,
            )
            if spectral_norm:
                fc = nn.utils.spectral_norm(fc, name='W')
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayer(
            ensemble_size, in_size, output_size,
            w_std_value=1/(2*np.sqrt(in_size)),
            b_init_value=b_init_value,
        )

    def forward(self, input):
        dim = len(input.shape)

        # input normalization
        h = (input - self.input_mu) / self.input_std

        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, output_size)
        return output

    def sample(self, input):
        preds = self.forward(input)

        inds = torch.randint(0, len(self.elites), input.shape[:-1])
        inds = inds.unsqueeze(dim=-1).to(device=self.device)
        inds = inds.repeat(1, preds.shape[2])

        samples = (inds == 0).float() * preds[self.elites[0]]
        for i in range(1, len(self.elites)):
            samples += (inds == i).float() * preds[self.elites[i]]

        return samples

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = torch.from_numpy(mean, device=self.device)
        self.input_std.data = torch.from_numpy(std, device=self.device)


class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        device,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        self.device = device

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim)).to(self.device)
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim), device=self.device, dtype=torch.float32)
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            device,
            init_w=3e-3,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
                device=device,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
            device=device,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                torch.nn.init.orthogonal_(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]
        
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output

    def sample(self, *inputs):
        preds = self.forward(*inputs)
        
        return torch.min(preds, dim=0)[0]

    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError
