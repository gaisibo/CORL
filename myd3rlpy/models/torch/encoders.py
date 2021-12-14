from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from d3rlpy.itertools import last_flag
from d3rlpy.torch_utility import View

from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction, _PixelEncoder, PixelEncoder, PixelEncoderWithAction, _VectorEncoder, VectorEncoder, VectorEncoderWithAction


class EncoderWithTaskID(Encoder, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        pass

    @property
    def task_id_size(self) -> int:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        pass


class EncoderWithActionWithTaskID(EncoderWithAction, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor, task_id: int) -> torch.Tensor:
        pass

    @property
    def task_id_size(self) -> int:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, action: torch.Tensor, task_id: int) -> torch.Tensor:
        pass


class PixelEncoderWithTaskID(_PixelEncoder, EncoderWithTaskID):

    _task_id_size: int
    _discrete_task_id: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        task_id_size: int,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_task_id: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._task_id_size = task_id_size
        self._discrete_task_id = discrete_task_id
        super().__init__(
            observation_shape=observation_shape,
            filters=filters,
            feature_size=feature_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
        )

    def _get_linear_input_size(self) -> int:
        size = super()._get_linear_input_size()
        return size + self._task_id_size

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._conv_encode(x)

        if self._discrete_task_id:
            task_id = F.one_hot(
                task_id.view(-1).long(), num_classes=self._task_id_size
            ).float()

        # cocat feature and task_id
        h = torch.cat([h.view(h.shape[0], -1), task_id], dim=1)
        h = self._activation(self._fc(h))
        if self._use_batch_norm:
            h = self._fc_bn(h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)

        return h

    @property
    def task_id_size(self) -> int:
        return self._task_id_size

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        modules: List[torch.nn.Module] = []

        # add linear layer
        in_features = self._fc.in_features - self._task_id_size
        modules.append(nn.Linear(self.get_feature_size(), in_features))
        modules.append(self._activation)

        # reshape output
        modules.append(View((-1, *self._get_last_conv_shape()[1:])))

        # add conv layers
        for is_last, conv in last_flag(reversed(self._convs)):
            deconv = nn.ConvTranspose2d(
                conv.out_channels,
                conv.in_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
            )
            modules.append(deconv)

            if not is_last:
                modules.append(self._activation)

        return modules


class PixelEncoderWithActionWithTaskID(_PixelEncoder, EncoderWithActionWithTaskID):

    _action_size: int
    _discrete_action: bool
    _task_id_size: int
    _discrete_task_id: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        task_id_size: int,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        self._task_id_size = task_id_size
        self._discrete_task_id = discrete_task_id
        super().__init__(
            observation_shape=observation_shape,
            filters=filters,
            feature_size=feature_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
        )

    def _get_linear_input_size(self) -> int:
        size = super()._get_linear_input_size()
        return size + self._action_size + self._task_id_size

    def forward(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._conv_encode(x)

        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()
        if self._discrete_task_id:
            task_id = F.one_hot(
                task_id.view(-1).long(), num_classes=self._task_id_size
            ).float()

        # cocat feature and action
        h = torch.cat([h.view(h.shape[0], -1), action, task_id], dim=1)
        h = self._activation(self._fc(h))
        if self._use_batch_norm:
            h = self._fc_bn(h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)

        return h

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def task_id_size(self) -> int:
        return self._task_id_size

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        modules: List[torch.nn.Module] = []

        # add linear layer
        in_features = self._fc.in_features - self._action_size - self._task_id_size
        modules.append(nn.Linear(self.get_feature_size(), in_features))
        modules.append(self._activation)

        # reshape output
        modules.append(View((-1, *self._get_last_conv_shape()[1:])))

        # add conv layers
        for is_last, conv in last_flag(reversed(self._convs)):
            deconv = nn.ConvTranspose2d(
                conv.out_channels,
                conv.in_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
            )
            modules.append(deconv)

            if not is_last:
                modules.append(self._activation)

        return modules

class VectorEncoderWithTaskID(_VectorEncoder, EncoderWithTaskID):

    _task_id_size: int
    _discrete_task_id: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        task_id_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        discrete_task_id: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._task_id_size = task_id_size
        self._discrete_task_id = discrete_task_id
        concat_shape = (observation_shape[0] + task_id_size,)
        super().__init__(
            observation_shape=concat_shape,
            hidden_units=hidden_units,
            use_batch_norm=use_batch_norm,
            use_dense=use_dense,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        if self._discrete_task_id:
            task_id = F.one_hot(
                task_id.view(-1).long(), num_classes=self.task_id_size
            ).float()
        x = torch.cat([x, task_id], dim=1)
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def task_id_size(self) -> int:
        return self._task_id_size

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        assert not self._use_dense, "use_dense=True is not supported yet"
        modules: List[torch.nn.Module] = []
        for is_last, fc in last_flag(reversed(self._fcs)):
            if is_last:
                in_features = fc.in_features - self._task_id_size
            else:
                in_features = fc.in_features

            modules.append(nn.Linear(fc.out_features, in_features))

            if not is_last:
                modules.append(self._activation)
        return modules


class VectorEncoderWithActionWithTaskID(_VectorEncoder, EncoderWithActionWithTaskID):

    _action_size: int
    _discrete_action: bool
    _task_id_size: int
    _discrete_task_id: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        task_id_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        discrete_action: bool = False,
        discrete_task_id: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        self._task_id_size = task_id_size
        self._discrete_task_id = discrete_task_id
        concat_shape = (observation_shape[0] + action_size + task_id_size,)
        super().__init__(
            observation_shape=concat_shape,
            hidden_units=hidden_units,
            use_batch_norm=use_batch_norm,
            use_dense=use_dense,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()
        if self._discrete_task_id:
            task_id = F.one_hot(
                task_id.view(-1).long(), num_classes=self.task_id_size
            ).float()
        x = torch.cat([x, action, task_id], dim=1)
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def task_id_size(self) -> int:
        return self._task_id_size

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        assert not self._use_dense, "use_dense=True is not supported yet"
        modules: List[torch.nn.Module] = []
        for is_last, fc in last_flag(reversed(self._fcs)):
            if is_last:
                in_features = fc.in_features - self._action_size - self._task_id_size
            else:
                in_features = fc.in_features

            modules.append(nn.Linear(fc.out_features, in_features))

            if not is_last:
                modules.append(self._activation)
        return modules
