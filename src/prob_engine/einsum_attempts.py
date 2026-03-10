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

class CollectSubsequences(torch.nn.Module):
    def __init__(self, size_out: int, device = None):
        super().__init__()
        assert size_out > 0
        self.size_out = size_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        size_in = input.shape[-1]
        assert size_in >= self.size_out
        if size_in == self.size_out:
            # Do nothing
            return input.unsqueeze(-2)
        elif self.size_out == 1:
            # Do nothing
            return input.unsqueeze(-1)
        else:
            indices = torch.arange(size_in)
            combs = torch.combinations(indices, r=self.size_out)
            return input[..., combs]

class tempPDF_32(torch.nn.Module):
    # (x,y,z) -> ((x,y), (y,z), (z,x)) -> A_{x,y}*B_{y,z}*C_{z,x}/tr(ABC)
    def __init__(self, size_out: int, resolution: int, device=None):
        super().__init__()
        assert size_out > 0 and resolution > 0
        self.size_in = 3
        self.size_out = size_out
        self.resolution = resolution

        self.weight = torch.nn.Parameter(
            torch.empty(
                [self.size_out, self.size_in] + [self.resolution]*2,
                device = device, dtype = torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def weight_measure(self):
        # tr(ABC) = \sum_{i,j,k} a_{i,j}b_{j,k}c_{k,i}
        result = torch.einsum('bij,bjk,bki->b', *self.weight.abs().unbind(1))
        return result
    
    def cell_values(self):
        result = torch.einsum('bij,bjk,bki->bijk', *self.weight.abs().unbind(1))
        weight_measure = torch.einsum("bijk->b", result)
        result = torch.einsum("b...,b->b...", result, 1.0/weight_measure)
        return result * math.pow(float(self.resolution),3)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (self.size_in,)
        w = self.weight.abs()
        w_m = self.weight_measure().pow(1.0/3.0).view(self.size_out,1,1,1)
        w *= (1.0/w_m)
        inds = input.view((batch_shape.numel(), self.size_in)) * self.resolution
        inds = torch.minimum(inds.floor(), torch.tensor([self.resolution-1]))\
            .to(dtype=torch.int64)
        batch_range = torch.arange(self.size_in)\
            .expand(batch_shape.numel(), self.size_in)
        selected = w.abs()[..., batch_range, inds, inds.roll(-1, -1)]
        result = selected.prod(-1)      #a_{x,y}b_{y,z}c_{z,x}
        result = torch.transpose(result,0,-1)
        return result.view(batch_shape + (self.size_out,))

class einsumPDF_32_A(torch.nn.Module):
    # (x,y,z) -> ((x,y), (y,z), (z,x)) -> A_{x,y}*B_{y,z}*C_{z,x}/tr(ABC)
    def __init__(self, size_out: int, resolution: int, device=None):
        super().__init__()
        assert size_out > 0 and resolution > 0
        self.size_in = 3
        self.size_out = size_out
        self.resolution = resolution

        self.weight = torch.nn.Parameter(
            torch.empty(
                [self.size_out, self.size_in] + [self.resolution]*2,
                device = device, dtype = torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def weight_measure(self):
        # tr(ABC) = \sum_{i,j,k} a_{i,j}b_{j,k}c_{k,i}
        result = torch.einsum('bij,bjk,bki->b', *self.weight.abs().unbind(1))
        return result
    
    def cell_values(self):
        result = torch.einsum('bij,bjk,bki->bijk', *self.weight.abs().unbind(1))
        weight_measure = torch.einsum("bijk->b", result)
        result = torch.einsum("b...,b->b...", result, 1.0/weight_measure)
        return result * math.pow(float(self.resolution),3)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (self.size_in,)
        w = self.weight.abs()
        w_m = self.weight_measure().pow(1.0/3.0).view(self.size_out,1,1,1)
        w *= (1.0/w_m)
        inds = input.view((batch_shape.numel(), self.size_in)) * self.resolution
        inds = torch.minimum(inds.floor(), torch.tensor([self.resolution-1])).to(dtype=torch.int64)
        ind_vecs = torch.nn.functional.one_hot(inds, self.resolution) # (B, 3, resolution)
        result = torch.einsum("B...i,b...ij->Bb...ij", ind_vecs, w)
        result = torch.einsum("Bb...ij,B...j->Bb...ij", result, ind_vecs[...,[1,2,0],:])
        result = torch.einsum("Bb...ij->Bb...", result)
        result = result.prod(-1)
        return result.view(batch_shape + (self.size_out,))

class einsumPDF_32_B(torch.nn.Module):
    # (x,y,z) -> ((x,y), (y,z), (z,x)) -> A_{x,y}*B_{y,z}*C_{z,x}/tr(ABC)
    def __init__(self, size_out: int, resolution: int, device=None):
        super().__init__()
        assert size_out > 0 and resolution > 0
        self.size_in = 3
        self.size_out = size_out
        self.resolution = resolution

        self.weight = torch.nn.Parameter(
            torch.empty(
                [self.size_out, self.size_in] + [self.resolution]*2,
                device = device, dtype = torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def weight_measure(self):
        # tr(ABC) = \sum_{i,j,k} a_{i,j}b_{j,k}c_{k,i}
        result = torch.einsum('bij,bjk,bki->b', *self.weight.abs().unbind(1))
        return result
    
    def cell_values(self):
        result = torch.einsum('bij,bjk,bki->bijk', *self.weight.abs().unbind(1))
        weight_measure = torch.einsum("bijk->b", result)
        result = torch.einsum("b...,b->b...", result, 1.0/weight_measure)
        return result * math.pow(float(self.resolution),3)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (self.size_in,)
        inds = input.view((batch_shape.numel(), self.size_in)) * self.resolution
        inds = torch.minimum(inds.floor(), torch.tensor([self.resolution-1])).to(dtype=torch.int64)
        ind_tensors = torch.nn.functional.one_hot(inds, self.resolution).to(dtype=torch.float32) # (B, 3, resolution)
        ind_tensors = torch.einsum("Bi,Bj,Bk->Bijk", *ind_tensors.unbind(-2))
        result = torch.einsum("Bijk,bijk->Bb",ind_tensors, self.cell_values())
        return result.view(batch_shape + (self.size_out,))

class einsumCDF_32_A(torch.nn.Module):
    # (x,y,z) -> ((x,y), (y,z), (z,x)) -> A_{x,y}*B_{y,z}*C_{z,x}/tr(ABC)
    def __init__(self, size_out: int, resolution: int, device=None):
        super().__init__()
        assert size_out > 0 and resolution > 0
        self.size_in = 3
        self.size_out = size_out
        self.resolution = resolution

        self.weight = torch.nn.Parameter(
            torch.empty(
                [self.size_out, self.size_in] + [self.resolution]*2,
                device = device, dtype = torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def weight_measure(self):
        # tr(ABC) = \sum_{i,j,k} a_{i,j}b_{j,k}c_{k,i}
        result = torch.einsum('bij,bjk,bki->b', *self.weight.abs().unbind(1))
        return result
    
    def cell_values(self):
        result = torch.einsum('bij,bjk,bki->bijk', *self.weight.abs().unbind(1))
        weight_measure = torch.einsum("bijk->b", result)
        result = torch.einsum("b...,b->b...", result, 1.0/weight_measure)
        return result * math.pow(self.resolution, 3)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (self.size_in,)
        w = self.weight.abs()
        w_m = self.weight_measure().pow(1.0/3.0).view(self.size_out,1,1,1)
        w *= (1.0/w_m)
        pts = input.view((batch_shape.numel(), self.size_in))
        inds = torch.minimum((pts * self.resolution).floor(),
                            torch.tensor([self.resolution-1])).to(dtype=torch.int64)
        fracts = torch.frac(pts * self.resolution) + (pts >= 1.0)
        hot = torch.nn.functional.one_hot(inds, self.resolution).to(dtype=torch.float32) # (B, 3, resolution)
        covered = (torch.arange(self.resolution) < inds.unsqueeze(-1)).int()
        measured = covered + (hot * fracts.unsqueeze(-1))
        #print("measured", measured)
        measured *= math.pow(self.resolution,-1/36)
        #Flawed calculations
        #do something like measured[0], w[0], measured[1], w[1], measured[2], w[2] instead?
        result = torch.einsum("B...i,b...ij,B...j->Bb...ij", measured, w, measured.roll(1,-2))
        result = torch.einsum("Bb...ij,Bb...jk,Bb...ki->Bb...", result[...,0,:,:], result[...,1,:,:], result[...,2,:,:])
        return result.view(batch_shape + (self.size_out,))

class einsumCDF_32_B(torch.nn.Module):
    def __init__(self, size_out: int, resolution: int, device=None):
        super().__init__()
        assert size_out > 0 and resolution > 0
        self.size_in = 3
        self.size_out = size_out
        self.resolution = resolution

        self.weight = torch.nn.Parameter(
            torch.empty(
                [self.size_out, self.size_in] + [self.resolution]*2,
                device = device, dtype = torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def weight_measure(self):
        # tr(ABC) = \sum_{i,j,k} a_{i,j}b_{j,k}c_{k,i}
        result = torch.einsum('bij,bjk,bki->b', *self.weight.abs().unbind(1))
        return result

    def cell_values(self):
        result = torch.einsum('bij,bjk,bki->bijk', *self.weight.abs().unbind(1))
        weight_measure = torch.einsum("bijk->b", result)
        result = torch.einsum("b...,b->b...", result, 1.0/weight_measure)
        return result * math.pow(self.resolution, 3)    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (self.size_in,)
        pts = input.view((batch_shape.numel(), self.size_in))
        inds = torch.minimum((pts * self.resolution).floor(),
                            torch.tensor([self.resolution-1])).to(dtype=torch.int64)
        fracts = torch.frac(pts * self.resolution) + (pts >= 1.0)
        hot = torch.nn.functional.one_hot(inds, self.resolution).to(dtype=torch.float32) # (B, 3, resolution)
        covered = (torch.arange(self.resolution) < inds.unsqueeze(-1)).int()
        measured = covered + (hot * fracts.unsqueeze(-1))
        #print("measured", measured)
        measured *= math.pow(self.resolution,-1)
        ind_tensors = torch.einsum("Bi,Bj,Bk->Bijk", *measured.unbind(-2))
        result = torch.einsum("Bijk,bijk->Bb",ind_tensors, self.cell_values())
        return result.view(batch_shape + (self.size_out,))

def einsum_test():
    import time
    from prob_engine.derivatives import get_partial_batched
    out, res = 1, 2
    c1, c2 = einsumCDF_32_B(out, res), einsumCDF_32_B(out, res)
    c2.weight = c1.weight
    print((c2.weight==c1.weight).all())
    pts = torch.cat((torch.rand(2,3), torch.tensor([[0.9999, 0.9999, 0.9999], [1.0,1.0,1.0], [0.0,0.0,0.0]])), 0)
    print(pts.shape)
    start = time.time()
    r1 = c1.forward(pts)
    end = time.time()
    print(end-start, "\n", r1)
    start = time.time()
    r2 = c2.forward(pts)
    end = time.time()
    print(end-start, "\n", r2)
    print("Hello")
    pts2_1 = torch.tensor([[0.0, 0.3, 0.5], [1.0,1.0,1.0], [0.0,0.0,0.0]]).requires_grad_(True)
    pts2_2 = torch.tensor([[0.0, 0.3, 0.5], [1.0,1.0,1.0], [0.0,0.0,0.0]]).requires_grad_(True)
    res1 = c1.forward(pts2_1)
    res2 = c2.forward(pts2_2)
    print("c1 density", "\n", res1.squeeze(0), get_partial_batched(pts2_1, res1.squeeze(0), [0,1,2]), "\n", c1.cell_values(), c1.cell_values().sum(), c1.weight_measure())
    print("c2 density", "\n", res2.squeeze(0), get_partial_batched(pts2_2, res2.squeeze(0), [0,1,2]), "\n", c2.cell_values(), c2.cell_values().sum(), c2.weight_measure())
    print("Finished einsum_test")

def einsum_comp_test():
    #do self-composed cdf function...
    #res = m.forward(stack((m1.forward(pts),m2.forward(pts),m3.forward(pts)),-1))
    #plot(partial(pts, res, [0,1,2])))
    from prob_engine.derivatives import get_partial_batched
    from matplotlib import pyplot
    cdf, cdf2 = einsumCDF_32_B(1, 10), einsumCDF_32_B(3, 10)
    def my_cdf(pts: torch.Tensor)->torch.Tensor:
        return cdf.forward(cdf2.forward(pts))
    min_bound, max_bound, bins = 0, 1, 60
    width = 1 / bins
    sample1 = torch.linspace(
        min_bound + 0.5 * width,
        max_bound - 0.5 * width,
        bins,
        dtype=torch.float32)
    sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
    sample2 = torch.stack(sample2, dim=-1).view(
        torch.Size((bins*bins, 2)))
    sample2 = torch.cat((sample2, 0.5*torch.ones(sample2.shape[:-1] + (1,))), -1).requires_grad_(True)
    value2 = my_cdf(sample2)
    deriv = get_partial_batched(sample2, value2, [0,1,2]).view(torch.Size((bins,bins)))
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        value2.view(torch.Size((bins,bins))).cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("CDF plot")
    pyplot.show()
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        deriv.cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("PDF plot")
    pyplot.show()
    print("Finished einsum_comp_test")

def einsum_training_test_image(path: str = "", train_steps: int = 10000, train_samples: int = 1000):
    from prob_engine.image_conversion import get_uniformgrid_from_image
    ug = get_uniformgrid_from_image(path)
    def target_cdf(input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (3,)
        #first coordinate is artificially added, as ug is defined on [0,1]^2
        #result will be scaled by the first coordinate, thus the slice
        #on {1}x[0,1]^2 will be the cdf of ug.
        result = ug.get_cdf(input[...,1:])*input[...,1]
        return result
    model = torch.nn.Sequential(
            einsumCDF_32_B(3, 10),
            einsumCDF_32_B(3, 10),
            einsumCDF_32_B(3, 10),
            #einsumCDF_32_B(3, 10),
            einsumCDF_32_B(1, 10),
        )
    def training(steps: int = 10000, samples: int = 1000) -> None:
        from prob_engine.derivatives import get_partial_batched
        from matplotlib import pyplot
        ug.plot_exact_pdf(0.0, 1.0, bins = min(ug._counts.max().item(), 512))
        opt = torch.optim.Adam(model.parameters(), lr = 1e-3)

        min_bound, max_bound, bins = 0, 1, 10
        width = 1 / bins
        sample1 = torch.linspace(
            min_bound + 0.5 * width,
            max_bound - 0.5 * width,
            bins,
            dtype=torch.float32)
        sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
        sample2 = torch.stack(sample2, dim=-1).view(
            torch.Size((bins*bins, 2)))
        sample2 = torch.cat((sample2, torch.ones(sample2.shape[:-1] + (1,))), -1)

        for step in range(steps):
            opt.zero_grad()
            sample = torch.rand((samples, 3), dtype = torch.float32)
            error = torch.pow(model.forward(sample) - target_cdf(sample), 2).sum()
            avg_error = error * (1.0/samples)
            print(avg_error)
            avg_error.backward()
            opt.step()
            if step % 500 == 0:
                sample22 = sample2.requires_grad_(True)
                value2 = model.forward(sample22)
                pyplot.pcolormesh(
                    sample1.cpu().numpy(),
                    sample1.cpu().numpy(),
                    value2.view(torch.Size((bins,bins))).cpu().detach().numpy(),
                    rasterized=True)
                pyplot.colorbar()
                pyplot.title("CDF plot")
                pyplot.show()
                deriv = get_partial_batched(sample22, value2, [0,1,2]).view(torch.Size((bins,bins)))
                pyplot.pcolormesh(
                    sample1.cpu().numpy(),
                    sample1.cpu().numpy(),
                    deriv.cpu().detach().numpy(),
                    rasterized=True)
                pyplot.colorbar()
                pyplot.title("PDF plot")
                pyplot.show()
        bins = 60
        sample22 = sample2.requires_grad_(True)
        value2 = model.forward(sample22)
        pyplot.pcolormesh(
            sample1.cpu().numpy(),
            sample1.cpu().numpy(),
            value2.view(torch.Size((bins,bins))).cpu().detach().numpy(),
            rasterized=True)
        pyplot.colorbar()
        pyplot.title("CDF plot")
        pyplot.show()
        deriv = get_partial_batched(sample22, value2, [0,1,2]).view(torch.Size((bins,bins)))
        pyplot.pcolormesh(
            sample1.cpu().numpy(),
            sample1.cpu().numpy(),
            deriv.cpu().detach().numpy(),
            rasterized=True)
        pyplot.colorbar()
        pyplot.title("PDF plot")
        pyplot.show()
    training(train_steps, train_samples)