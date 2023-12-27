# Copyright (C) 2023, Miklos Maroti
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
import torch
from typing import Iterator, List, Optional

from .distribution import Distribution


class PosLinearLayer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, device=None):
        super().__init__()

        self.size_in = size_in
        self.size_out = size_out

        self.weight = torch.nn.Parameter(
            torch.empty((size_in, size_out), device=device, dtype=torch.float32))
        self.bias = torch.nn.Parameter(
            torch.empty((size_out,), device=device, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.size_in)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.abs(self.weight)
        temp = torch.matmul(input, weight)
        return torch.add(temp, self.bias)


class MinMaxLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[-1] % 2 == 0
        half = x.shape[-1] // 2
        a = torch.narrow(x, dim=-1, start=0, length=half)
        b = torch.narrow(x, dim=-1, start=half, length=half)
        c = torch.minimum(a, b)
        d = torch.maximum(a, b)
        e = torch.cat((c, d), dim=-1)
        assert e.shape == x.shape
        return e


class Relu2Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.relu(x)
        b = -torch.relu(-x)
        c = torch.cat((a, b), dim=-1)
        return c


class NeuralNet(Distribution):
    def __init__(self,
                 layers: List[torch.nn.Module],
                 bounds: torch.Tensor,
                 device: Optional[str] = None):
        assert bounds.ndim == 2 and bounds.shape[0] == 2

        Distribution.__init__(self, bounds.shape[1:], device=device)
        self._bounds = bounds.to(dtype=torch.float32, device=self._device)

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        return self._bounds[0]

    @property
    def max_bounds(self) -> torch.Tensor:
        return self._bounds[1]
