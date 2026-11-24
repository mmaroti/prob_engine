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

import torch
from matplotlib import pyplot
from prob_engine.uniform_grid import UniformGrid
from prob_engine.derivatives import get_partial_batched

def test1():
    g = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    min_bound, max_bound, bins = 0, 1, 60
    g.plot_exact_pdf(0, 1)
    g.plot_exact_cdf(0, 1)
    width = 1 / bins
    sample1 = torch.linspace(
        min_bound + 0.5 * width,
        max_bound - 0.5 * width,
        bins,
        dtype=torch.float32)
    sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
    sample2 = torch.stack(sample2, dim=-1).view(
        torch.Size((bins*bins, 2))).requires_grad_(True)
    value2 = g.get_cdf(sample2)
    deriv = get_partial_batched(sample2, value2, [0,1]).view(torch.Size((bins,bins)))
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
    print("Finished test1")

def test2():
    g = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g1 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g2 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    min_bound, max_bound, bins = 0, 1, 60
    width = 1 / bins
    sample1 = torch.linspace(
        min_bound + 0.5 * width,
        max_bound - 0.5 * width,
        bins,
        dtype=torch.float32)
    sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
    sample2 = torch.stack(sample2, dim=-1).view(
        torch.Size((bins*bins, 2))).requires_grad_(True)
    val1 = torch.stack([g1.get_cdf(sample2), g2.get_cdf(sample2)], -1)
    val2 = g.get_cdf(val1)
    der = get_partial_batched(sample2, val2, [0,1]).view(torch.Size((bins,bins)))
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        val2.view(torch.Size((bins,bins))).cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("CDF plot")
    pyplot.show()
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        der.cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("PDF plot")
    pyplot.show()
    print("Finished test2")

def test3():
    g = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([10,10]))
    g1 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([10,10]))
    g2 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([10,10]))
    min_bound, max_bound, bins = 0, 1, 60
    width = 1 / bins
    sample1 = torch.linspace(
        min_bound + 0.5 * width,
        max_bound - 0.5 * width,
        bins,
        dtype=torch.float32)
    sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
    sample2 = torch.stack(sample2, dim=-1).view(
        torch.Size((bins*bins, 2))).requires_grad_(True)
    val1 = torch.stack([g1.get_cdf(sample2), g2.get_cdf(sample2)], -1)
    val2 = g.get_cdf(val1)
    der = get_partial_batched(sample2, val2, [0,1]).view(torch.Size((bins,bins)))
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        val2.view(torch.Size((bins,bins))).cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("CDF plot")
    pyplot.show()
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        der.cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("PDF plot")
    pyplot.show()
    print("Finished test3")

def test4():
    g = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g1 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g2 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g11 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g12 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g21 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    g22 = UniformGrid(torch.tensor([[0,0],[1,1]]), torch.tensor([5,5]))
    min_bound, max_bound, bins = 0, 1, 60
    width = 1 / bins
    sample1 = torch.linspace(
        min_bound + 0.5 * width,
        max_bound - 0.5 * width,
        bins,
        dtype=torch.float32)
    sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
    sample2 = torch.stack(sample2, dim=-1).view(
        torch.Size((bins*bins, 2))).requires_grad_(True)
    val1 = g1.get_cdf(torch.stack([g11.get_cdf(sample2), g12.get_cdf(sample2)], -1))
    val2 = g2.get_cdf(torch.stack([g21.get_cdf(sample2), g22.get_cdf(sample2)], -1))
    val = g.get_cdf(torch.stack([val1,val2], -1))
    der = get_partial_batched(sample2, val, [0,1]).view(torch.Size((bins,bins)))
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        val.view(torch.Size((bins,bins))).cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("CDF plot")
    pyplot.show()
    pyplot.pcolormesh(
        sample1.cpu().numpy(),
        sample1.cpu().numpy(),
        der.cpu().detach().numpy(),
        rasterized=True)
    pyplot.colorbar()
    pyplot.title("PDF plot")
    pyplot.show()
    print("Finished test4")