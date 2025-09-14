# Copyright (C) 2023, Miklos Maroti and Daniel Bezdany
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
from scipy.special import comb
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


class PosQuadraticLayer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, device=None):
        super().__init__()

        assert size_in > 0 and size_out > 0
        self.size_in = size_in
        self.size_out = size_out

        self.weight1 = torch.nn.Parameter(
            torch.empty((size_in, size_out),
                        device=device, dtype=torch.float32))
        self.weight2 = torch.nn.Parameter(
            torch.empty((size_in, size_in, size_out),
                        device=device, dtype=torch.float32))
        self.bias = torch.nn.Parameter(
            torch.empty((size_out,),
                        device=device, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.size_in)
        torch.nn.init.uniform_(self.weight1, -bound, bound)
        torch.nn.init.uniform_(self.weight2, -bound, bound)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        temp = self.weight1.abs()
        temp = torch.matmul(input, temp)
        result = temp
        temp = self.weight2.abs()
        temp = torch.einsum("...bi,jik->...bjk", input, temp)
        temp = torch.einsum("...bi,...bki->...bk", input, temp)
        result = torch.add(result, temp)
        result = torch.add(result, self.bias)
        return result

class PosPolynomLayer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, degree: int, device = None):
        super().__init__()

        assert size_in > 0 and size_out > 0
        assert degree > 0
        self.size_in = size_in
        self.size_out = size_out
        self.degree = degree
        #TODO check that this structure interacts nicely with module.parameters
        self.bias = torch.nn.Parameter(
            torch.empty((size_out,), device=device, dtype=torch.float32))
        combination_counts = [int(comb(size_in+i-1, i, exact = True)) 
                              for i in range(1, self.degree+1)]
        self.weights = [torch.nn.Parameter(
                        torch.empty( (size_out, c),
                            device=device, dtype=torch.float32) )
                        for c in combination_counts]
        for i, w in enumerate(self.weights):
            name = "weight_"+str(i+1)
            self.register_parameter(name, w)
        #self.register_parameter("bias", self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.size_in)
        if self.degree > 0:
            for w in self.weights:
                torch.nn.init.uniform_(w, 0, bound)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.size_in
        batch_shape = input.shape[:-1]
        input = input.view((batch_shape.numel(),input.shape[-1]))
        result = torch.zeros((input.shape[0],1),
                             dtype=torch.float32,
                             device=self.bias.device)
        result = torch.add(result, self.bias)
        for i, w in enumerate(self.weights):
            combos = torch.combinations(
                torch.arange(0, self.size_in),
                r = i+1, with_replacement=True)
            temp = input[:,combos].prod(-1).unsqueeze(-2)
            temp = torch.mul(w,temp)
            result = torch.add(result, temp.sum(-1))
        return result.view(batch_shape + (self.size_out,))

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


class ProductLayer(torch.nn.Module):
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
        torch.nn.init.normal_(self.weight, 0, 1)
        torch.nn.init.uniform_(self.bias, 0, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert torch.all(0.0 < input)
        # Needs 2 to ensure that the second derivatives are positive
        weight = self.weight.abs() + 2
        temp = torch.matmul(input.log(), weight).exp()
        return torch.mul(temp, self.bias.abs())


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


class NeuralDist(Distribution):
    def __init__(self, device: Optional[str] = None):
        super().__init__(event_size=1, device=device)

        self.model = NormalizerLayer(torch.nn.Sequential(
            PosLinearLayer(1, 50),
            # MinMaxLayer(),
            Relu2Layer(),
            PosLinearLayer(100, 50),
            # MinMaxLayer(),
            Relu2Layer(),
            PosLinearLayer(100, 1),
            ExponentialLayer(),
        ))

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-1]
        assert sample.shape == batch_shape + self.event_shape
        sample = sample.to(dtype=torch.float32, device=self._device)
        sample = sample.reshape(batch_shape + (self._event_size, ))

        result = self.model.forward(sample).squeeze(-1)
        assert result.shape == batch_shape
        return result


def test():
    input = torch.linspace(0.0, 1.0, 101, dtype=torch.float32)
    expected = torch.sqrt(input)

    model = NeuralDist()
    optim = torch.optim.Adam(model.parameters, lr=1e-3)
    loss = torch.nn.MSELoss()

    for step in range(10001):
        optim.zero_grad()

        output = model.get_cdf(input)
        error = loss(output, expected)
        if step % 1000 == 0:
            print(step, error.detach().cpu().item())

        error.backward()
        optim.step()

    model.plot_exact_cdf()
