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

from typing import Optional
import torch

from prob_engine.distribution import Distribution
from prob_engine.uniform_grid import UniformGrid


class UniformGridShell(UniformGrid):
    def __init__(self,
                 center: torch.Tensor,
                 radius1: torch.Tensor,
                 radius2: torch.Tensor,
                 counts: torch.Tensor,
                 use_exact_rectangle_probs: bool = False,
                 device: Optional[str] = None):
        
        assert center.dim() == 1
        assert radius1.numel() == 1 and radius2.numel() == 1
        assert (radius1.count_nonzero().item() \
                + radius2.count_nonzero().item() > 0)
        assert center.shape == counts.shape
        r = torch.minimum(radius1.abs(), radius2.abs())
        R = torch.maximum(radius1.abs(), radius2.abs())
        bounds = torch.stack((center - R, center + R), 0)
        UniformGrid.__init__(self, bounds, counts, device)
        if use_exact_rectangle_probs:
            if self._event_size > 2:
                raise NotImplementedError()
            from prob_engine.misc.uniform_shell import UniformShell
            UniformGrid.__init__(self, bounds, counts, device)
            ShellDist = UniformShell(center, radius1, radius2, device)
            self.initialize_from_distribution_pdf_rectangle(ShellDist)
        else:
            centers = self.centers().view(self._counts.prod(), self._event_size)
            inside_R = (centers - center.view((self._event_size,))
                        ).pow(2).sum(-1) <= R.pow(2).item()
            outside_r = (centers - center.view((self._event_size,))
                         ).pow(2).sum(-1) >= r.pow(2).item()
            correct = torch.logical_and(inside_R, outside_r).to(
                dtype=torch.float32, device=self._device)
            self._parameter = torch.nn.Parameter(
                correct.view(self._parameter.shape))


def test():
    from prob_engine.misc.uniform_shell import UniformShell

    center1 = torch.tensor([0.25])
    radius1 = torch.tensor(0.1)
    Radius1 = torch.tensor(0.25)
    counts1 = torch.tensor([10])

    dist1 = UniformGridShell(center1, radius1, Radius1, counts1)
    print("Parameters", list(dist1.parameters))
    dist1.plot_exact_pdf()
    dist1.plot_empirical_pdf()
    dist1.plot_exact_cdf()
    dist1.plot_empirical_cdf()

    center2 = torch.tensor([0.0,0.0])
    radius2 = torch.tensor(0.2)
    Radius2 = torch.tensor(0.5)
    counts2 = torch.tensor([10,10])

    dist2 = UniformGridShell(center2,radius2,Radius2,counts2,False)
    distShell = UniformShell(center2, radius2, Radius2)
    print("Parameters", list(dist2.parameters))
    dist2.plot_exact_pdf()
    distShell.plot_exact_pdf()
    dist2.plot_empirical_pdf()
    distShell.plot_empirical_pdf()
    dist2.plot_exact_cdf()
    distShell.plot_exact_cdf()
    dist2.plot_empirical_cdf()
    distShell.plot_empirical_cdf()

    dist3 = UniformGridShell(center2,radius2,Radius2,counts2,True)
    print("Parameters", list(dist3.parameters))
    dist3.plot_exact_pdf()
    distShell.plot_exact_pdf()
    dist3.plot_empirical_pdf()
    distShell.plot_empirical_pdf()
    dist3.plot_exact_cdf()
    distShell.plot_exact_cdf()
    dist3.plot_empirical_cdf()
    distShell.plot_empirical_cdf()