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
from typing import Iterator, Optional

from .distribution import Distribution


class PosLinearLayer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, device=None):
        super().__init__()

        assert size_in > 0 and size_out > 0
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] % 2 == 0
        half = input.shape[-1] // 2
        input1 = torch.narrow(input, dim=-1, start=0, length=half)
        input2 = torch.narrow(input, dim=-1, start=half, length=half)
        minimum = torch.minimum(input1, input2)
        maximum = torch.maximum(input1, input2)
        output = torch.cat((minimum, maximum), dim=-1)
        assert output.shape == input.shape
        return output


class Relu2Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        positive = torch.relu(input)
        negative = -torch.relu(-input)
        output = torch.cat((positive, negative), dim=-1)
        return output


class ExponentialLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(input)


class NormalizerLayer(torch.nn.Module):
    def __init__(self, child: torch.nn.Module):
        super().__init__()
        self.child = child

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert torch.all(-1.0 <= input) and torch.all(input <= 1.0)

        base = torch.full((input.shape[-1], ), fill_value=-1.0,
                          device=input.device, dtype=input.dtype)

        value0 = self.child(base)
        value1 = self.child(torch.ones_like(base))
        value2 = self.child(input)
        output = (value2 - value0) / (value1 - value0)
        assert output.shape == value2.shape
        return output


class NeuralNet(Distribution):
    def __init__(self, device: Optional[str] = None):
        super().__init__(event_shape=torch.Size([1]), device=device)

        self.model = NormalizerLayer(torch.nn.Sequential(
            PosLinearLayer(1, 4),
            MinMaxLayer(),
            PosLinearLayer(4, 1),
            ExponentialLayer(),
        ))

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        sample_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == sample_shape + self.event_shape
        sample = sample.to(dtype=torch.float32, device=self._device)
        sample = sample.reshape(sample_shape + (self.event_numel, ))

        result = self.model.forward(sample).squeeze(-1)
        assert result.shape == sample_shape
        return result


def test():
    dist = NeuralNet()
    dist.plot_exact_cumulative()
