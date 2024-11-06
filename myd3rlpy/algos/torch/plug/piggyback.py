import numpy as np
import torch
from myd3rlpy.algos.torch.plug.plug import Plug


class Piggyback(Plug):
    def __init__(self, ratio: float, algo, networks, smallest_threshold = 10) -> None:
        self._ratio = ratio
        self._algo = algo
        self._networks = networks
        self._smallest_threshold = smallest_threshold

    def soft_init(self, soft_network, piggyback_dim):

    def build(self):
        self._piggyback_params = []
        for network in self._networks:
            self._piggyback_params.append([])
            for pp in network.parameters():
                if pp.numel() > self._smallest_threshold:
                    self._piggyback_params[-1].append(pp)
        self.piggyback_dims = [[pp.data.numel() for pp in piggyback_params] for piggyback_params in self._piggyback_params]
        self.soft_networks = [dict() for _ in self.piggyback_dims]
        self.copy_networks = [dict() for _ in self.piggyback_dims]
        self.masks = [dict() for _ in self.piggyback_dims]

    def _soft_mask_networks(self, network, piggyback_dim, soft_param, copy_param):
        count = 0
        masks = []
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
                masks.append(mask)
        return torch.concatenate(masks, dim=0)

    def _pre_piggyback_loss(self):
        for network_id, (network, soft_networks, copy_networks, masks, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.copy_networks, self.masks, self.piggyback_dims)):
            if self._algo._impl_id not in soft_networks.keys():
                soft_param = torch.zeros(np.sum(piggyback_dim))
                torch.nn.init.normal_(soft_param)
                soft_param = torch.nn.Parameter(soft_param)
                self.soft_networks[network_id][self._algo._impl_id] = soft_param.to(self._networks[0].device)
                self.masks[network_id][self._algo._impl_id] = self._soft_mask_networks(network, piggyback_dim, soft_param, self.copy_networks[network_id][self._algo._impl_id])
            else:
                soft_param = self.soft_networks[network_id][self._algo._impl_id]

    def _post_piggyback_loss(self):
        for network_id, (network, soft_networks, copy_networks, masks, piggyback_dim) in enumerate(zip(self._networks, self.soft_networks, self.copy_networks, self.masks, self.piggyback_dims)):
            if self._algo._impl_id not in soft_networks.keys():
