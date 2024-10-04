from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
from d3rlpy.torch_utility import _WithDeviceAndScalerProtocol, _convert_to_torch
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.dataset import TransitionMiniBatch as OldTransitionMiniBatch
from myd3rlpy.dataset import TransitionMiniBatch
from d3rlpy.torch_utility import TorchMiniBatch as OldTorchMiniBatch


class TorchMiniBatch(OldTorchMiniBatch):
    def __init__(
        self,
        batch: TransitionMiniBatch,
        device: str,
        scaler: Optional[Scaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ):
        # convert numpy array to torch tensor
        observations = _convert_to_torch(batch.observations, device)
        actions = _convert_to_torch(batch.actions, device)
        rewards = _convert_to_torch(batch.rewards, device)
        if hasattr(batch, "rtgs"):
            rtgs = _convert_to_torch(batch.rtgs, device)
        next_observations = _convert_to_torch(batch.next_observations, device)
        if hasattr(batch, "next_actions"):
            next_actions = _convert_to_torch(batch.next_actions, device)
        terminals = _convert_to_torch(batch.terminals, device)
        n_steps = _convert_to_torch(batch.n_steps, device)

        # apply scaler
        if scaler:
            observations = scaler.transform(observations)
            next_observations = scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
            next_actions = action_scaler.transform(next_actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)
            if hasattr(batch, "rtgs"):
                rtgs = reward_scaler.transform(rtgs)

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        if hasattr(batch, "rtgs"):
            self._rtgs = rtgs
        self._next_observations = next_observations
        if hasattr(batch, "next_actions"):
            self._next_actions = next_actions
        self._terminals = terminals
        self._n_steps = n_steps
        self._device = device

    @property
    def rtgs(self) -> torch.Tensor:
        return self._rtgs

def torch_api(
    scaler_targets: Optional[List[str]] = None,
    action_scaler_targets: Optional[List[str]] = None,
    reward_scaler_targets: Optional[List[str]] = None,
) -> Callable[..., np.ndarray]:
    def _torch_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        # get argument names
        sig = signature(f)
        arg_keys = list(sig.parameters.keys())[1:]

        def wrapper(
            self: _WithDeviceAndScalerProtocol, *args: Any, **kwargs: Any
        ) -> np.ndarray:
            tensors: List[Union[torch.Tensor, TorchMiniBatch]] = []

            # convert all args to torch.Tensor
            for i, val in enumerate(args):
                tensor: Union[torch.Tensor, TorchMiniBatch]
                if isinstance(val, torch.Tensor):
                    tensor = val
                elif isinstance(val, list):
                    tensor = default_collate(val)
                    tensor = tensor.to(self.device)
                elif isinstance(val, np.ndarray):
                    if val.dtype == np.uint8:
                        dtype = torch.uint8
                    else:
                        dtype = torch.float32
                    tensor = torch.tensor(
                        data=val,
                        dtype=dtype,
                        device=self.device,
                    )
                elif val is None:
                    tensor = None
                elif isinstance(val, TransitionMiniBatch) or isinstance(val, OldTransitionMiniBatch):
                    tensor = TorchMiniBatch(
                        val,
                        self.device,
                        scaler=self.scaler,
                        action_scaler=self.action_scaler,
                        reward_scaler=self.reward_scaler,
                    )
                else:
                    tensor = torch.tensor(
                        data=val,
                        dtype=torch.float32,
                        device=self.device,
                    )

                if isinstance(tensor, torch.Tensor):
                    # preprocess
                    if self.scaler and scaler_targets:
                        if arg_keys[i] in scaler_targets:
                            tensor = self.scaler.transform(tensor)

                    # preprocess action
                    if self.action_scaler and action_scaler_targets:
                        if arg_keys[i] in action_scaler_targets:
                            tensor = self.action_scaler.transform(tensor)

                    # preprocessing reward
                    if self.reward_scaler and reward_scaler_targets:
                        if arg_keys[i] in reward_scaler_targets:
                            tensor = self.reward_scaler.transform(tensor)

                    # make sure if the tensor is float32 type
                    if tensor is not None and tensor.dtype != torch.float32:
                        tensor = tensor.float()

                tensors.append(tensor)
            return f(self, *tensors, **kwargs)

        return wrapper

    return _torch_api
