import copy
from typing import Optional, Sequence, List, cast, Union

import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api, to_cpu, to_cuda, get_state_dict, set_state_dict, map_location
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.dataset import TransitionMiniBatch

from myd3rlpy.models.vaes import create_vae_factory, VAEFactory
from myd3rlpy.models.torch.vaes import VAE
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class StateVAEImpl:

    _learning_rate: float
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _factory: VAEFactory
    _use_gpu: Optional[Device]
    _vae: Optional[VAE]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        learning_rate: float,
        optim_factory: OptimizerFactory,
        factory: VAEFactory,
        replay_type: str,
        replay_lambda: float,
        gem_alpha: float,
        agem_alpha: float,
        ewc_rwalk_alpha: float,
        use_gpu: Optional[Device],
    ):
        self._observation_shape = observation_shape
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._factory = factory
        self._replay_type = replay_type
        self._replay_lambda = replay_lambda
        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._use_gpu = use_gpu

        # initialized in build
        self._vae = None
        self._optim = None

    def build(self) -> None:
        # setup torch models
        self._build_vae()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

        assert self._vae is not None
        if self._replay_type in ['ewc', 'rwalk', 'si']:
            # Store current parameters for the next task
            self._older_params = {n: p.clone().detach() for n, p in self._vae.named_parameters() if p.requires_grad}
            if self._replay_type in ['ewc', 'rwalk']:
                # Store fisher information weight importance
                self._fisher = {n: torch.zeros(p.shape).to(self._device) for n, p in self._vae.named_parameters() if p.requires_grad}
            if self._replay_type == 'rwalk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._W = {n: torch.zeros(p.shape).to(self._device) for n, p in self._vae.named_parameters() if p.requires_grad}
                self._scores = {n: torch.zeros(p.shape).to(self._device) for n, p in self._vae.named_parameters() if p.requires_grad}
            elif self._replay_type == 'si':
                self._W = {n: p.clone().detach().zero_() for n, p in self._vae.named_parameters() if p.requires_grad}
                self._omega = {n: p.clone().detach().zero_() for n, p in self._vae.named_parameters() if p.requires_grad}
        elif self._replay_type == 'gem':
            # Allocate temporary synaptic memory
            self._grad_dims = []
            for pp in self._vae.parameters():
                self._grad_dims.append(pp.data.numel())
            self._grads_cs = {}
            self._grads_da = torch.zeros(np.sum(self._grad_dims)).to(self._device)
        elif self._replay_type == 'agem':
            self._grad_dims = []
            for param in self._vae.parameters():
                self._grad_dims.append(param.data.numel())
            self._grad_xy = torch.Tensor(np.sum(self._grad_dims)).to(self._device)
            self._grad_er = torch.Tensor(np.sum(self._grad_dims)).to(self._device)

    def _build_vae(self) -> None:
        self._vae = self._factory.create(self._observation_shape)

    def _build_optim(self) -> None:
        assert self._vae is not None
        self._optim = self._optim_factory.create(
            self._vae.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api()
    def update_vae(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._optim is not None
        assert self._vae is not None

        self._optim.zero_grad()

        loss = self._compute_loss(batch.observations)

        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    def _compute_loss(self, x: torch.Tensor):
        assert self._vae is not None
        recon_x, mu, logvar = self._vae(x)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def to_gpu(self, device: Device = Device()) -> None:
        self._device = f"cuda:{device.get_id()}"
        to_cuda(self, self._device)

    def to_cpu(self) -> None:
        self._device = "cpu:0"
        to_cpu(self)

    def save_model(self, fname: str) -> None:
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

    @train_api
    def update(self, batch_tran: TransitionMiniBatch, replay_batch: Optional[List[torch.Tensor]]=None):
        batch = TorchMiniBatch(
            batch_tran,
            self._device,
            scaler=None,
            action_scaler=None,
            reward_scaler=None,
        )
        assert self._optim is not None
        assert self._vae is not None

        unreg_grads = None
        curr_feat_ext = None

        loss = 0
        replay_loss = 0
        if replay_batch is not None:
            replay_loss = 0
            replay_batch = [x.to(self._device) for x in replay_batch]
            replay_batch = dict(zip(replay_name[:-2], replay_batch))
            replay_batch = Struct(**replay_batch)
            if self._replay_type == "orl":
                replay_orl_loss = self._compute_loss(replay_batch.observations)
                replay_loss = replay_loss + replay_orl_loss
            elif self._replay_type == 'generate':
                generated_x = self._clone_vae.generate(batch.observations.shape[0], batch.observations)
                replay_orl_loss = self._compute_loss(generated_x)
                replay_loss = replay_loss + replay_orl_loss
            elif self._replay_type == "ewc":
                replay_loss_ = 0
                for n, p in self._vae.named_parameters():
                    if n in self._fisher.keys():
                        replay_loss = replay_loss + torch.mean(self._fisher[n] * (p - self._older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_loss_
            elif self._replay_type == 'rwalk':
                curr_feat_ext = {n: p.clone().detach() for n, p in self._vae.named_parameters() if p.requires_grad}
                # store gradients without regularization term
                unreg_grads = {n: p.grad.clone().detach() for n, p in self._vae.named_parameters()
                               if p.grad is not None}

                self._optim.zero_grad()
                # Eq. 3: elastic weight consolidation quadratic penalty
                replay_loss_ = 0
                for n, p in self._vae.named_parameters():
                    if n in self._fisher.keys():
                        replay_loss_ = replay_loss_ + torch.mean((self._fisher[n] + self._scores[n]) * (p - self._older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_loss_
            elif self._replay_type == 'si':
                for n, p in self._vae.named_parameters():
                    if p.grad is not None and n in self._fisher.keys():
                        self._W[n].add_(-p.grad * (p.detach() - self._older_params[n]))
                    self._older_params[n] = p.detach().clone()
                replay_loss_ = 0
                for n, p in self._vae.named_parameters():
                    if p.requires_grad:
                        replay_loss_ = replay_loss_ + torch.mean(self._omega[n] * (p - self._older_params[n]) ** 2)
                replay_loss = replay_loss + replay_loss_
            elif self._replay_type == 'gem':
                replay_batch = cast(TorchMiniBatch, replay_batch)
                replay_loss_ = self._compute_loss(replay_batch.observations)
                replay_loss = replay_loss_
                replay_loss.backward()
                store_grad(self._vae.parameters, self._grads_cs[i], self._grad_dims)
            elif self._replay_type == "agem":
                store_grad(self._vae.parameters, self._grad_xy, self._grad_dims)
                replay_loss_ = self._compute_loss(replay_batch.observations)
                replay_loss = replay_loss + replay_loss_

        self._optim.zero_grad()
        loss = self._compute_loss(batch.observations)
        if self._replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc']:
            loss = loss + self._replay_lambda * replay_loss
        loss.backward()
        if replay_batch is not None:
            if self._replay_type == 'agem':
                replay_loss.backward()
                store_grad(self._vae.parameters, self._grad_er, self._grad_dims)
                dot_prod = torch.dot(self._grad_xy, self._grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self._grad_xy, ger=self._grad_er)
                    overwrite_grad(self._vae.parameters, g_tilde, self._grad_dims)
                else:
                    overwrite_grad(self._vae.parameters, self._grad_xy, self._grad_dims)
            elif self._replay_type == 'gem':
                # copy gradient
                store_grad(self._vae.parameters, self._grads_da, self._grad_dims)
                dot_prod = torch.mm(self._grads_da.unsqueeze(0),
                                torch.stack(list(self._grads_cs).values()).T)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self._grads_da.unsqueeze(1),
                                  torch.stack(list(self._grads_cs).values()).T, margin=self._gem_alpha)
                    # copy gradients back
                    overwrite_grad(self._vae.parameters, self._grads_da,
                                   self._grad_dims)
        self._optim.step()

        if replay_batch is not None:
            if self._replay_type == 'rwalk':
                assert unreg_grads is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    for n, p in self._vae.named_parameters():
                        if n in unreg_grads.keys():
                            self._W[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

        loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss

    def save_clone_data(self):
        assert self._vae is not None
        self._clone_vae = copy.deepcopy(self._vae)
