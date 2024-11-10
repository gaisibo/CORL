import numpy as np
import torch
from myd3rlpy.algos.torch.plug.plug import Plug


class Piggyback(Plug):
    def __init__(self, algo, networks, ratio: float = 10, smallest_threshold: int = 10) -> None:
        super().__init__(algo, networks)
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
        self.soft_networks = []
        self.copy_networks = []
        self.usable_networks = []
        for network_id, (network, piggyback_dim) in enumerate(self._networks, self.piggyback_dims):
            soft_param = torch.zeros(np.sum(piggyback_dim))
            copy_param = torch.zeros(np.sum(piggyback_dim)).to(self._networks[0].device)
            usable_param = torch.ones(np.sum(piggyback_dim)).to(self._networks[0].device)
            torch.nn.init.normal_(soft_param)
            soft_param = torch.nn.Parameter(soft_param)
            self.soft_networks[network_id] = soft_param.to(self._networks[0].device)
            self._algo.critic_optim.add_param_group({"params": self.soft_networks[network_id], "lr": 0.001})
            self.copy_networks[network_id] = copy_param
            self.usable_networks[network_id] = usable_param
        self.masks = [dict() for _ in self.piggyback_dims]
        #self.soft_optimizer = algo.

    # Change the param into mask * param
    def _pre_soft_mask_networks(self, network, piggyback_dim, soft_param, usable_param, copy_param):
        count = 0
        soft_param = soft_param * usable_param
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(self.piggyback_dims[:count + 1])
                soft_param_pp = soft_param[begin: end]
                pp_num = pp.numel()
                assert pp_num == end - begin
                pp_ratio_num = pp_num * self._ratio
                soft_threshold = soft_param_pp.topk(pp_ratio_num)[-1]
                mask = torch.threshold(soft_param_pp, soft_threshold, 0)
                # 毁在这里复制进去，然后在后面拿出来。
                copy_param[begin: end].copy_(pp.data)
                pp.data.copy_(mask)
                mask[mask > 0] = 1
                #masks.append(mask)
        #return torch.concatenate(masks, dim=0)

    def _pre_mask_networks(self, network, piggyback_dim, copy_param, mask):
        count = 0
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(self.piggyback_dims[:count + 1])
                copy_param[begin: end].copy_(pp.data)
                pp.data.copy_(pp.data * mask)

    # Change the mask * param into param, and assign the grad for masks and params
    def _post_soft_mask_networks(self, network, piggyback_dim, soft_param, usable_param, copy_param, mask):
        count = 0
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(self.piggyback_dims[:count + 1])
                pp_num = pp.numel()
                assert pp_num == end - begin
                time_grad = pp.grad
                pp.data.copy_(copy_param[begin: end])
                pp.grad.copy_(time_grad * mask[begin: end])
                if self._new_task == "new":
                    soft_param[begin: end].copy_(time_grad * pp.data * usable_param[begin: end])
                else:
                    soft_param[begin: end].zeros_()

    # Change the param into mask * param
    def _post_task_networks(self, network, piggyback_dim, soft_param, usable_param):
        count = 0
        masks = []
        soft_param = soft_param * usable_param
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(self.piggyback_dims[:count + 1])
                soft_param_pp = soft_param[begin: end]
                pp_num = pp.numel()
                assert pp_num == end - begin
                pp_ratio_num = pp_num * self._ratio
                soft_threshold = soft_param_pp.topk(pp_ratio_num)[-1]
                mask = torch.threshold(soft_param_pp, soft_threshold, 0)
                mask[mask > 0] = 1
                masks.append(mask)
        mask_param = torch.concatenate(masks, dim=0)
        usable_param = torch.where(mask_param == 1, torch.zeros_like(usable_param), usable_param)
        return mask_param, usable_param

    # Change the mask * param into param, and assign the grad for masks and params
    def _post_mask_networks(self, network, piggyback_dim, copy_param, mask):
        count = 0
        masks = []
        for pp in network.parameters():
            if pp.numel() > self._smallest_threshold:
                begin = 0 if count == 0 else sum(piggyback_dim[:count])
                end = np.sum(self.piggyback_dims[:count + 1])
                pp_num = pp.numel()
                assert pp_num == end - begin
                pp.data.copy_(copy_param[begin: end])

    def _pre_piggyback_loss(self):
        if self._new_task:
            for _, (network, soft_param, usable_param, copy_param, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.copy_networks, self.piggyback_dims)):
                #self.masks[network_id][self._algo._impl_id] = self._pre_soft_mask_networks(network, piggyback_dim, soft_param, self.copy_networks[network_id][self._algo._impl_id])
                self._pre_soft_mask_networks(network, piggyback_dim, soft_param, usable_param, copy_param)
        else:
            for _, (network, masks, copy_param, piggyback_dim) in enumerate(zip(self._networks, self.masks, self.copy_networks, self.piggyback_dims)):
                self._pre_mask_networks(network, piggyback_dim, copy_param, masks[self._algo._impl_id])

    def _post_piggyback_loss(self):
        for _, (network, soft_param, usable_param, copy_param, masks, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.copy_networks, self.masks, self.piggyback_dims)):
            self._post_soft_mask_networks(network, piggyback_dim, soft_param, usable_param, copy_param, masks[self._algo._impl_id])

    def _pre_piggyback_task(self):
        if self._algo._impl_id not in self.masks[0].keys():
            self._new_task = True
        else:
            self._new_task = False

    def _post_piggyback_task(self):
        if self._new_task:
            self._new_task = False
            for network_id, (network, soft_param, usable_param, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.usable_networks, self.piggyback_dims)):
                self.masks[network_id][self._algo._impl_id] = self._post_task_networks(network, piggyback_dim, soft_param, usable_param)

    def _pre_piggyback_evaluation(self):
        assert len(self.masks) > 0
        assert self._algo._impl_id in self.masks[0].keys()
        for _, (network, copy_param, masks, piggyback_dim) in enumerate(zip(self._networks, self.copy_networks, self.masks, self.piggyback_dims)):
            self._pre_mask_networks(network, piggyback_dim, copy_param, masks[self._algo._impl_id])

    def _post_piggyback_evaluation(self):
        assert len(self.masks) > 0
        assert self._algo._impl_id in self.masks[0].keys()
        for _, (network, copy_param, masks, piggyback_dim) in enumerate(zip(self._networks, self.copy_networks, self.masks, self.piggyback_dims)):
            self._post_mask_networks(network, piggyback_dim, copy_param, masks[self._algo._impl_id])
