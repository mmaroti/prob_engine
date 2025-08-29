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
from typing import Iterator, Optional, Callable, Any, Dict
from prob_engine.distribution import Distribution
from prob_engine.uniform_grid import UniformGrid
import time

def multilinspace_old(start: torch.Tensor, 
                  end: torch.Tensor,
                  steps: int) -> torch.Tensor:
    """
    Generates points from multi dimensional rectangle [start, end],
    each axis being split into 'steps' many sections.
    """
    assert start.shape == end.shape
    flat_start = start.flatten()
    flat_end = end.flatten()
    arrays = []
    for i in range(0,start.numel()):
        arrays += [torch.linspace(flat_start[i], flat_end[i], steps)]
    return torch.cartesian_prod(*arrays)

def multilinspace(start: torch.Tensor, 
                  end: torch.Tensor,
                  steps: torch.Tensor) -> torch.Tensor:
    """
    Generates points from multi dimensional rectangle [start, end],
    the i-th axis being split into 'steps[i-1]' many sections.
    """
    assert start.shape == end.shape and steps.shape == start.shape
    flat_start = start.flatten()
    flat_end = end.flatten()
    flat_steps = steps.flatten()
    my_arrays = []
    for i in range(0,start.numel()):
        my_arrays.append(torch.linspace(
            flat_start[i], flat_end[i], 
            int(flat_steps[i])))
    return torch.cartesian_prod(*my_arrays)

def cdf_cdf_Lp(dist: Distribution,
                     target: Distribution,
                     points: torch.Tensor,
                     p: float) -> torch.Tensor:
    """
    Calculates L_p distance
    between 'dist' cdf and 'target' empirical cdf
    using values evaluated at 'points'. 
    """
    assert dist.event_shape == target.event_shape
    assert points.shape[-len(dist.event_shape):] == dist.event_shape

    y1 = dist.get_cdf(points)
    y2 = target.get_cdf(points)
    result = (y1-y2).abs().pow(p).sum(-1).pow(1.0/p)
    return result

def cdf_ecdf_Lp(dist: Distribution,
                     target: Distribution,
                     sample_count: int,
                     points: torch.Tensor,
                     p: float) -> torch.Tensor:
    """
    Calculates L_p distance
    between 'dist' cdf and 'target' empirical cdf
    using values evaluated at 'points'. 
    """
    assert dist.event_shape == target.event_shape
    assert points.shape[-len(dist.event_shape):] == dist.event_shape

    y1 = dist.get_cdf(points)
    y2 = target.get_empirical_cdf(sample_count, points)
    result = (y1-y2).abs().pow(p).sum(-1).pow(1.0/p)
    return result

def cdf_cdf_E_Lp(dist: Distribution,
                     target: Distribution,
                     bounds: torch.Tensor,
                     E_count: int,
                     sampling_count: int,
                     p: float) -> torch.Tensor:
    """
    Calculates expected value of L_p distance
    between 'dist' cdf and 'target' empirical cdf
    evaluated at random points from within 'bounds'. 
    """
    assert dist.event_shape == target.event_shape
    assert bounds.shape[0] == 2 and bounds.shape[-len(dist.event_shape):] == dist.event_shape

    total = 0.0
    generator = UniformGrid(bounds, torch.ones(bounds.shape[1:]))
    for i in range(0,E_count):
        pts = generator.sample(torch.Size([sampling_count]))
        y1 = dist.get_cdf(pts)
        y2 = target.get_cdf(pts)
        total += (y1-y2).abs().pow(p).sum().pow(1.0/p)
    return total/float(E_count)

def cdf_ecdf_E_Lp(dist: Distribution,
                     target: Distribution,
                     bounds: torch.Tensor,
                     E_count: int,
                     sampling_count: int,
                     empirical_count: int,
                     p: float) -> torch.Tensor:
    """
    Calculates expected value of L_p distance
    between 'dist' cdf and 'target' empirical cdf
    evaluated at random points from within 'bounds'. 
    """
    assert dist.event_shape == target.event_shape
    assert bounds.shape[0] == 2 and bounds.shape[1:] == dist.event_shape

    total = 0.0
    generator = UniformGrid(bounds, torch.ones(bounds.shape[1:]))
    for i in range(0,E_count):
        pts = generator.sample((sampling_count,))
        y1 = dist.get_cdf(pts)
        y2 = target.get_empirical_cdf(empirical_count, pts)
        total += (y1-y2).abs().pow(p).sum().pow(1.0/p)
    return total/float(E_count)

def train_distribution(
        dist: Distribution, target: Distribution, steps: int, 
        loss_func: Callable[[Distribution, Distribution], torch.Tensor]):
    #TODO: Allow for other kinds of optimization
    opt = torch.optim.Adam(dist.parameters, lr = 1e-3)
    error_tracker = []
    start = time.time()
    for step in range(steps):
        opt.zero_grad()
        error = loss_func(dist, target)
        if step % 100 == 0:
            end = time.time()
            print("step:", step, "error:", error.detach().cpu().item(), "time since last print", end-start, "sec")
            start = end
            error_tracker.append(error.item())
        if step % 1000 == 0:
            dist.plot_exact_cdf()
            dist.plot_exact_pdf()
        error.backward()
        opt.step()
    animate_datalist(error_tracker)

def animate_datalist(data: list):
    import matplotlib.pyplot as plt
    plt.ion()

    y = [data[0]]
    x = [0]

    graph = plt.plot(x,y)[0]
    plt.ylim(0,max(data)*1.2)
    #plt.xlim(0,len(data))
    pause_time = min(5.0/len(data),0.01)
    print(pause_time, "seconds")

    for d in data[1:]:
        y.append(d)
        x.append(x[-1]+1)
        
        graph.remove()
        graph = plt.plot(x,y,color = 'b')[0]
        plt.xlim(x[0], x[-1])
        plt.pause(pause_time)
    plt.show(block = True)
