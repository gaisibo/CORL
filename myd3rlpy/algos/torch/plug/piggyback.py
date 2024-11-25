import numpy as np
import torch
from myd3rlpy.algos.torch.plug.plug import Plug


class Piggyback(Plug):
    def __init__(self, algo, networks, update, optim, ratio: float = 0.1, smallest_threshold: int = 10) -> None:
        super().__init__(algo, networks, update, optim)
        self._ratio = ratio
        self._smallest_threshold = smallest_threshold

    def build(self):
        self._piggyback_params = []
        for network in self._networks:
            self._piggyback_params.append([])
            for pp in network.parameters():
                if pp.numel() > self._smallest_threshold:
                    self._piggyback_params[-1].append(pp)
        self.piggyback_dims = [[pp.data.numel() for pp in piggyback_params] for piggyback_params in self._piggyback_params]
        self.soft_networks = torch.nn.ParameterList()
        for network_id, (network, piggyback_dim) in enumerate(zip(self._networks, self.piggyback_dims)):
            soft_param = torch.zeros(np.sum(piggyback_dim))
            soft_param = soft_param.to(self.device)
            torch.nn.init.normal_(soft_param)
            soft_param = torch.nn.Parameter(soft_param)
            #soft_param.grad = torch.zeros_like(soft_param.data)
            self.soft_networks.append(soft_param)
            self._algo._critic_optim.add_param_group({"params": self.soft_networks[network_id], "lr": 0.001})
        with torch.no_grad():
            self.copy_networks = torch.nn.ParameterList()
            self.usable_networks = torch.nn.ParameterList()
            for network_id, (network, piggyback_dim) in enumerate(zip(self._networks, self.piggyback_dims)):
                copy_param = torch.zeros(np.sum(piggyback_dim)).to(self.device)
                usable_param = torch.ones(np.sum(piggyback_dim)).to(self.device)
                self.copy_networks.append(copy_param)
                self.usable_networks.append(usable_param)
            self.masks = torch.nn.ModuleList()
            self.masks.append(torch.nn.ParameterDict())
        #self.masks = [dict() for _ in self.piggyback_dims]
        #self.soft_optimizer = algo.

    # Change the param into mask * param
    def _pre_soft_mask_networks(self, network, piggyback_dim, soft_param, usable_param, copy_param):
        count = 0
        soft_param = soft_param * usable_param
        masks = []
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(piggyback_dim[:count + 1])
                soft_param_pp = soft_param[begin: end]
                pp_num = pp.numel()
                assert pp_num == end - begin
                pp_ratio_num = int(pp_num * self._ratio)
                soft_threshold = soft_param_pp.topk(pp_ratio_num)[0][-1].item()
                mask = torch.threshold(soft_param_pp, soft_threshold, 0)
                # 在这里复制进去，然后在后面拿出来。
                copy_param.data[begin: end].copy_(pp.data.reshape(-1))
                pp.data.copy_(mask.reshape(pp.shape))
                mask[mask > 0] = 1
                count += 1
                masks.append(mask)
        return torch.concatenate(masks, dim=0)

    def _pre_mask_networks(self, network, piggyback_dim, copy_param, mask, evaluation=False):
        count = 0
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(piggyback_dim[:count + 1])
                copy_param.data[begin: end].copy_(pp.data.reshape(-1))
                pp.data.copy_(pp.data * (mask[begin: end].reshape(pp.shape)))
                if count == 0 and evaluation:
                    print(f"evaluation {self._algo._impl_id}")
                    print(f"pp.numel(): {pp.data.numel()}")
                    print(f"pp.sum(): {torch.sum(pp.data)}")
                    print(f"pp.data > 0: {torch.sum(pp.data > 0)}")
                    print(f"pp.data = 0: {torch.sum(pp.data == 0)}")
                count += 1

    # Change the mask * param into param, and assign the grad for masks and params
    def _post_soft_mask_networks(self, network, piggyback_dim, soft_param, usable_param, copy_param, mask):
        count = 0
        soft_param.grad = torch.zeros_like(soft_param.data)
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(piggyback_dim[:count + 1])
                pp_num = pp.numel()
                assert pp_num == end - begin
                time_grad = pp.grad
                pp.data.copy_(copy_param[begin: end].reshape(pp.shape))
                pp.grad.copy_(time_grad * mask[begin: end].reshape(pp.shape))
                mask_usable = mask[begin: end] * torch.where(usable_param[begin: end] == 0, 1, 0)
                assert torch.sum(mask_usable) == 0
                if self._new_task:
                    soft_param.grad[begin: end].copy_((time_grad * pp.data).reshape(-1) * usable_param[begin: end])
                else:
                    soft_param.grad[begin: end].zeros_()
                count += 1

    # Change the mask * param into param, and assign the grad for masks and params
    def _post_soft_mask_networks(self, network, piggyback_dim, soft_param, usable_param, copy_param, mask):
        count = 0
        soft_param.grad = torch.zeros_like(soft_param.data)
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(piggyback_dim[:count + 1])
                pp_num = pp.numel()
                assert pp_num == end - begin
                time_grad = pp.grad
                pp.data.copy_(copy_param[begin: end].reshape(pp.shape))
                pp.data = torch.where(usable_param[begin: end].reshape(pp.shape) == 1, pp.data, copy_param[begin:end].reshape(pp.shape))
                count += 1

    # Change the param into mask * param
    def _post_task_networks(self, network, piggyback_dim, soft_param, usable_param):
        count = 0
        masks = []
        soft_param_bias = torch.min(soft_param)
        soft_param_biased = soft_param + soft_param_bias
        soft_param_biased = soft_param_biased * usable_param
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(piggyback_dim[:count + 1])
                soft_param_pp = soft_param_biased[begin: end]
                pp_num = pp.numel()
                assert pp_num == end - begin
                pp_ratio_num = int(pp_num * self._ratio)
                soft_threshold = soft_param_pp.topk(pp_ratio_num)[0][-1].item()
                mask = torch.threshold(soft_param_pp, soft_threshold, 0)
                mask[mask > 0] = 1
                masks.append(mask)
                if count == 0:
                    pp = pp.reshape(-1) * mask
                    print(f"post_network {self._algo._impl_id}")
                    print(f"pp.numel(): {pp.data.numel()}")
                    print(f"pp.sum(): {torch.sum(pp.data)}")
                    print(f"pp.data > 0: {torch.sum(pp.data > 0)}")
                    print(f"pp.data = 0: {torch.sum(pp.data == 0)}")
                count += 1
        mask_param = torch.concatenate(masks, dim=0)
        usable_param = torch.where(mask_param == 1, torch.zeros_like(usable_param), usable_param)
        return mask_param, usable_param

    # Change the mask * param into param, and assign the grad for masks and params
    def _post_mask_networks(self, network, piggyback_dim, copy_param):
        count = 0
        masks = []
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(piggyback_dim[:count + 1])
                pp_num = pp.numel()
                assert pp_num == end - begin
                pp.data.copy_(copy_param[begin: end].reshape(pp.shape))
                count += 1

    def pre_loss(self):
        if self._new_task:
            for network_id, (network, soft_param, usable_param, copy_param, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.copy_networks, self.piggyback_dims)):
                self.masks[network_id][str(self._algo._impl_id)] = self._pre_soft_mask_networks(network, piggyback_dim, soft_param, usable_param, copy_param)
                #self._pre_soft_mask_networks(network, piggyback_dim, soft_param, usable_param, copy_param)
        else:
            for _, (network, masks, copy_param, piggyback_dim) in enumerate(zip(self._networks, self.masks, self.copy_networks, self.piggyback_dims)):
                self._pre_mask_networks(network, piggyback_dim, copy_param, masks[str(self._algo._impl_id)])

    def post_loss(self):
        for _, (network, soft_param, usable_param, copy_param, masks, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.copy_networks, self.masks, self.piggyback_dims)):
            self._post_soft_mask_networks(network, piggyback_dim, soft_param, usable_param, copy_param, masks[str(self._algo._impl_id)])

    def post_step(self):
        for _, (network, soft_param, usable_param, copy_param, masks, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.copy_networks, self.masks, self.piggyback_dims)):
            self._post_soft_mask_networks(network, piggyback_dim, soft_param, usable_param, copy_param, masks[str(self._algo._impl_id)])

    def pre_task(self):
        if self._algo._impl_id not in self.masks[0].keys():
            self._new_task = True
            self.hh = 1
        else:
            self._new_task = False
            self.hh = 2

    def post_task(self):
        if self._new_task:
            self._new_task = False
            for network_id, (network, soft_param, usable_param, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.piggyback_dims)):
                self.masks[network_id][str(self._algo._impl_id)], usable_param = self._post_task_networks(network, piggyback_dim, soft_param, usable_param)

    def pre_evaluation(self):
        assert len(self.masks) > 0
        assert str(self._algo._impl_id) in self.masks[0].keys()
        for _, (network, copy_param, masks, piggyback_dim) in enumerate(zip(self._networks, self.copy_networks, self.masks, self.piggyback_dims)):
            self._pre_mask_networks(network, piggyback_dim, copy_param, masks[str(self._algo._impl_id)], evaluation=True)

    def post_evaluation(self):
        for _, (network, copy_param, piggyback_dim) in enumerate(zip(self._networks, self.copy_networks, self.piggyback_dims)):
            self._post_mask_networks(network, piggyback_dim, copy_param)
