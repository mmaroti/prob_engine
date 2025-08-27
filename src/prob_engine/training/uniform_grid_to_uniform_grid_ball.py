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

from prob_engine.distribution import Distribution
from prob_engine.uniform_grid import UniformGrid
from prob_engine.testers.uniform_grid_ball import UniformGridBall
from prob_engine.training.training_misc import train_distribution, cdf_cdf_Lp, cdf_cdf_E_Lp, multilinspace

count = 20

center = torch.tensor([0.25,0.25])
radius = torch.tensor([0.5])
target_dist = UniformGridBall(center, radius, torch.tensor([30,30]), True)
dist = UniformGrid(torch.tensor([[-1,-1],[1,1]]), torch.tensor([count,count]))
compare_dist = UniformGrid(torch.tensor([[-1,-1],[1,1]]), torch.tensor([count,count]))
compare_dist.initialize_from_distribution_pdf_rectangle(target_dist)
target_dist.plot_exact_pdf()
compare_dist.plot_exact_pdf()
points = multilinspace(torch.tensor([0.0,0.0]), torch.tensor([1.0,1.0]), torch.tensor([100,100]))
def my_loss(d1: Distribution, d2: Distribution)->torch.Tensor:
    return cdf_cdf_Lp(d1, d2, points, 2)
print("Error between target and target-initialized distribution:",
      my_loss(compare_dist, target_dist).max().item())
#train_distribution(dist, target_dist, 2000, my_loss)
dist.plot_exact_pdf()
compare_dist.plot_exact_pdf()
bounds = torch.tensor([[-1,-1],[1,1]])
def my_loss2(d1: Distribution, d2: Distribution)->torch.Tensor:
    return cdf_cdf_E_Lp(d1, d2, bounds, 20, 500, 2)
dist2 = UniformGrid(torch.tensor([[-1,-1],[1,1]]), torch.tensor([count,count]))
train_distribution(dist2, target_dist, 3000, my_loss2)
dist2.plot_exact_pdf()
target_dist.plot_exact_pdf()
compare_dist.plot_exact_pdf()


