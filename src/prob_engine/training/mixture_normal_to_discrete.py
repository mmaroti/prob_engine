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
from prob_engine.discrete import Discrete
from prob_engine.mixture_normal import MixtureNormal
from prob_engine.training.training_misc import train_distribution, cdf_cdf_Lp, cdf_cdf_E_Lp, multilinspace

atom_count = 50
atoms = torch.rand((atom_count,2))*2-torch.tensor([1.0,1.0])
means = torch.rand((atom_count,2))*2-torch.tensor([1.0,1.0])
sdevs = torch.ones(means.shape)
target_dist = Discrete(atoms)
dist = MixtureNormal(means, sdevs*0.2)
compare_dist = MixtureNormal(means, sdevs)
compare_dist.initialize_from_distribution(target_dist)
target_dist.plot_exact_cdf()
compare_dist.plot_exact_cdf()
points = multilinspace(torch.tensor([0.0,0.0]), torch.tensor([1.0,1.0]), torch.tensor([100,100]))
def my_loss(d1: Distribution, d2: Distribution)->torch.Tensor:
    return cdf_cdf_Lp(d1, d2, points, 2)
print("Error between target and target-initialized distribution:",
      my_loss(compare_dist, target_dist).max().item())
train_distribution(dist, target_dist, 10000, my_loss)
dist.plot_exact_cdf()
compare_dist.plot_exact_cdf()
bounds = torch.tensor([[-1,-1],[1,1]])
def my_loss2(d1: Distribution, d2: Distribution)->torch.Tensor:
    return cdf_cdf_E_Lp(d1, d2, bounds, 20, 500, 2)
dist2 = MixtureNormal(means, sdevs*0.5)
train_distribution(dist2, target_dist, 10000, my_loss2)

