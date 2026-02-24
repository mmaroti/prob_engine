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

class einsumDensity(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, resolution: int, device=None):
        super().__init__()
        assert size_in > 0 and size_out > 0 and resolution > 0
        assert size_in == 3 and size_out == 1       #Temporary simplifying assumptions
        self.size_in = size_in
        self.size_out = size_out
        self.resolution = resolution

        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.size_out, self.size_in, self.resolution, self.resolution),
                device = device, dtype = torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def weight_measure_1(self):
        # \sum_{i,j,k} a_{i,j}b_{j,k}c_{k,i}
        # tr(ABC)
        A = self.weight[0,0,:]
        B = self.weight[0,1,:]
        C = self.weight[0,2,:]
        AB = torch.einsum('ij,jk->ik', A, B)
        ABC = torch.einsum('ik,kl->il', AB, C)
        result = torch.einsum("ii", ABC)
        print(result)
        return result
    
    def weight_measure_2(self):
        # \sum_{i,j,k} a_{i,j}b_{j,k}c_{i,k}
        # tr((AB)^T C)
        A = self.weight[0,0,:]
        B = self.weight[0,1,:]
        C = self.weight[0,2,:]
        AB = torch.einsum('ij,jk->ik', A, B)
        result = torch.einsum('ik,ik', AB, C)
        print(result)
        return result
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.size_in
        assert len(input.shape) == 2    #Assume there is a single batch dimension
        inds = input * self.resolution
        inds = torch.minimum(inds.floor(),
                torch.tensor([self.resolution-1])
                ).to(dtype=torch.int64)
        batch_range = torch.arange(
            self.size_in).expand(input.shape[0], self.size_in)
        selected = self.weight.squeeze(0)[batch_range, inds, inds.roll(-1, dims=-1)]
        result = selected.prod(-1)      #a_{x,y}b_{y,z}c_{z,x}
        norm_factor = (1.0/self.weight_measure_1().item())
        result *= norm_factor
        return result.unsqueeze(0)

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
        output = torch.add(temp, self.bias)
        return output


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
        output = temp
        temp = self.weight2.abs()
        temp = torch.einsum("...bi,jik->...bjk", input, temp)
        temp = torch.einsum("...bi,...bki->...bk", input, temp)
        output = torch.add(output, temp)
        output = torch.add(output, self.bias)
        return output

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
        output = torch.zeros((input.shape[0],1),
                             dtype=torch.float32,
                             device=self.bias.device)
        output = torch.add(output, self.bias)
        for i, w in enumerate(self.weights):
            #could use torch.outer or torch.einsum in for loop
            combos = torch.combinations(
                torch.arange(0, self.size_in),
                r = i+1, with_replacement=True)
            temp = input[:,combos].prod(-1).unsqueeze(-2)
            temp = torch.mul(w.abs(),temp).sum(-1)
            output = torch.add(output, temp)
        output = output.view(batch_shape + (self.size_out,))
        return output

class PosConvexLayer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, device=None):
        super().__init__()

        assert size_in > 0 and size_out > 0
        self.size_in = size_in
        self.size_out = size_out

        self.weight = torch.nn.Parameter(
            torch.empty((size_in, size_out), device=device, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.size_in)
        torch.nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.abs(self.weight)
        weight *= 1.0/weight.sum(-2)
        output = torch.matmul(input.squeeze(-1).squeeze(-1), weight)
        return output

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

    def forward(self, input) -> torch.Tensor:
        positive = torch.relu(input)
        negative = -torch.relu(-input)
        output = torch.cat((positive, negative), dim=-1)
        return output


class UniformMixLayer(torch.nn.Module):
    """Maps, coordinate-wise, into [0,1]"""
    def __init__(self, count: int, device = None):
        super().__init__()

        assert count > 0
        self.count = count
        self.bases = torch.nn.Parameter(
            torch.empty((count,), device=device, dtype=torch.float32))
        self.slopes = torch.nn.Parameter(
            torch.empty((count,), device=device, dtype=torch.float32))
        self.weights = torch.nn.Parameter(
            torch.empty((count,), device=device, dtype=torch.float32))
        #self.register_parameter("bias", self.bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.uniform_(self.bases, -1, 1)
        torch.nn.init.uniform_(self.slopes, 0.5, 2)
        bounds = 1.0/math.sqrt(self.count)
        torch.nn.init.uniform_(self.weights, -bounds, bounds)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(-1)
        output = (input - self.bases)*self.slopes
        output = torch.clamp(output, min=0.0, max=1.0)
        w = self.weights.abs()
        w *= 1.0/w.sum()
        output = (output*w).sum(-1)
        output = output.view(input.shape)
        return output

class SmoothCompactTransitionMixLayer(torch.nn.Module):
    """Maps, coordinate-wise, into [0,1]"""
    def __init__(self, count: int, device = None):
        super().__init__()

        assert count > 0
        self.count = count
        self.bases = torch.nn.Parameter(
            torch.empty((count,), device=device, dtype=torch.float32))
        self.slopes = torch.nn.Parameter(
            torch.empty((count,), device=device, dtype=torch.float32))
        self.weights = torch.nn.Parameter(
            torch.empty((count,), device=device, dtype=torch.float32))
        #self.register_parameter("bias", self.bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.uniform_(self.bases, -1, 1)
        torch.nn.init.uniform_(self.slopes, 0, 2)
        bounds = 1.0/math.sqrt(self.count)
        torch.nn.init.uniform_(self.weights, -bounds, bounds)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        input = input.view((batch_shape.numel(),input.shape[-1], 1))
        output = (input - self.bases)*self.slopes
        value1 = torch.exp(-1.0/output.relu())
        value2 = torch.exp(-1.0/(1.0-output).relu())
        output = value1/(value1+value2)
        output = (output*self.weights.abs()).sum(-1)
        output *= 1.0/self.weights.abs().sum()
        output = output.view(input.shape)
        return output
    
class AtanIdLayer(torch.nn.Module):
    """Maps, coordinate-wise, into [0,1], using arctan."""
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.atan(input)*(1/torch.pi)+0.5
        return output

class SigmoidLayer(torch.nn.Module):
    """Maps, coordinate-wise, into [0,1], using sigmoid function."""
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.sigmoid(input)
        return output

class TrigDuplicationLayer(torch.nn.Module):
    """Replaces x by sin(x + d), sin(2x + 2d), sin(3x+3d), etc."""
    def __init__(self, duplicates: int, device=None):
        super().__init__()
        self.duplicates = duplicates
    
    def forward(self, input: torch.Tensor)->torch.Tensor:
        scale = torch.arange(1, self.duplicates+1, 1,
                            dtype=torch.float32, device=input.device)
        shift = torch.e * torch.arange(1, self.duplicates+1, 1,
                            dtype=torch.float32, device=input.device)
        input_shape = input.shape
        temp = input.unsqueeze(-1) * scale + shift
        output = torch.sin(temp).view(input_shape[:-1] + (input_shape[-1]*temp.shape[-1],))
        return output

class NormalDuplicationLayer(torch.nn.Module):
    """Replaces x by the evaluation of shifted
       normal distribution CDFs at x"""
    def __init__(self, duplicates: int, device=None):
        super().__init__()
        self.duplicates = duplicates
    
    def forward(self, input: torch.Tensor)->torch.Tensor:
        means = (1/float(self.duplicates))*torch.arange(1, self.duplicates+1, 1,
                            dtype=torch.float32, device=input.device)
        vars = 1/math.sqrt(float(self.duplicates))
        input_shape = input.shape
        temp = input.unsqueeze(-1)
        sq2 = torch.tensor(2, device=input.device).sqrt()
        arg = temp - means
        arg = arg / (vars * sq2)
        result = 0.5 + 0.5 * torch.erf(arg)
        output = result.view(input_shape[:-1] + (input_shape[-1]*result.shape[-1],))
        return output

class ExponentialLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.exp(input)
        return output


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
        output = torch.mul(temp, self.bias.abs())
        return output


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

class ClampLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.clamp(input, min = 0.0, max = 1.0)
        return output

class TensorOutputLayer(torch.nn.Module):
    def __init__(self, child: torch.nn.Module):
        super().__init__()
        self.child = child

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.child(input)
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
