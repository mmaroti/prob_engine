# Copyright (C) 2023, Daniel Bezdany
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

from typing import Iterator, Optional
import torch

from prob_engine.distribution import Distribution
from prob_engine.uniform_grid import UniformGrid
from prob_engine.testers.uniform_ball import UniformBall

class UniformGridBall(UniformGrid):
    def __init__(self,
                 center: torch.Tensor,
                 radius: torch.Tensor,
                 counts: torch.Tensor,
                 device: Optional[str] = None):
        
        assert radius.numel() == 1
        assert (radius >= 0).all()
        bounds = torch.stack((center - radius.abs(), center + radius.abs()), 0)
        print(bounds.shape, counts.shape)
        UniformGrid.__init__(self, bounds, counts, device)
        BallDist = UniformBall(center, radius, device)
        self.initialize_from_distribution_pdf_rectangle(BallDist)

def test():
    dist1 = UniformGridBall(torch.tensor([0.25]), torch.tensor(0.25), torch.tensor([10]))
    print("Parameters", list(dist1.parameters))
    dist1.plot_exact_pdf()
    dist1.plot_empirical_pdf()
    dist1.plot_empirical_cdf()
    dist1.plot_exact_cdf()

    dist2 = UniformGridBall(torch.tensor([-0.5, -0.5]), torch.tensor(0.5), torch.tensor([10, 10]))
    distBall = UniformBall(torch.tensor([-0.5, -0.5]), torch.tensor(0.5))
    print("Parameters", list(dist2.parameters))
    dist2.plot_exact_pdf()
    distBall.plot_empirical_pdf()
    dist2.plot_empirical_pdf()
    dist2.plot_empirical_cdf()
    dist2.plot_exact_cdf()
    distBall.plot_empirical_cdf()

