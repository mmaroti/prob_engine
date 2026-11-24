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

from prob_engine.uniform_grid import UniformGrid
from prob_engine.normal import Normal

class UniformGridNormal(UniformGrid):
    def __init__(self,
                 means: torch.Tensor,
                 sdevs: torch.Tensor,
                 bounds: torch.Tensor,
                 counts: torch.Tensor,
                 use_exact_rectangle_probs: bool = False,
                 device: Optional[str] = None):
        
        assert bounds.shape[0] == 2 and bounds.shape[1:] == counts.shape
        assert means.shape == sdevs.shape and means.shape == counts.shape
        UniformGrid.__init__(self, bounds, counts, device)
        normal = Normal(means, sdevs, device)
        if use_exact_rectangle_probs:
            self.initialize_from_distribution_pdf_rectangle(normal)
        else:
            self.initialize_from_distribution_pdf_center(normal)

def test():
    means1 = torch.tensor([0.0])
    sdevs1 = torch.tensor([0.5])
    bounds1 = torch.tensor([[-0.75], [0.75]])
    counts1 = torch.tensor([20])

    ugn1 = UniformGridNormal(means1, sdevs1, bounds1, counts1, False)
    n1 = Normal(means1, sdevs1)
    ugn1.plot_exact_pdf()
    n1.plot_exact_pdf()
    ugn1.plot_empirical_pdf()
    n1.plot_empirical_pdf()
    ugn1.plot_exact_cdf()
    n1.plot_exact_cdf()
    ugn1.plot_empirical_cdf()
    ugn1.plot_empirical_cdf()

    ugn2 = UniformGridNormal(means1, sdevs1, bounds1, counts1, True)
    n2 = Normal(means1, sdevs1)
    ugn2.plot_exact_pdf()
    n2.plot_exact_pdf()
    ugn2.plot_empirical_pdf()
    n2.plot_empirical_pdf()
    ugn2.plot_exact_cdf()
    n2.plot_exact_cdf()
    ugn2.plot_empirical_cdf()
    ugn2.plot_empirical_cdf()

    means2 = torch.tensor([0.0, 0.0])
    sdevs2 = torch.tensor([0.5, 0.5])
    bounds2 = torch.tensor([[-0.75, -0.75],[0.75, 0.75]])
    counts2 = torch.tensor([20,20])

    ugn3 = UniformGridNormal(means2, sdevs2, bounds2, counts2, False)
    n3 = Normal(means2, sdevs2)
    ugn3.plot_exact_pdf()
    n3.plot_exact_pdf()
    ugn3.plot_empirical_pdf()
    n3.plot_empirical_pdf()
    ugn3.plot_exact_cdf()
    n3.plot_exact_cdf()
    ugn3.plot_empirical_cdf()
    ugn3.plot_empirical_cdf()

    print("Evaluation of get_cdf at Infinity:",
          ugn3.get_cdf(torch.full(ugn3.event_shape,torch.inf)))
    print("Evaluation of get_cdf at [0,Infinity]:",
          ugn3.get_cdf(torch.tensor([0.0,torch.inf])))
    print("CDF of marginal belonging to first coordinate at 0:",
          ugn3.get_cdf_marginal(torch.tensor([1,0]),torch.tensor([[0.0]])))

    ugn4 = UniformGridNormal(means2, sdevs2, bounds2, counts2, True)
    n4 = Normal(means2, sdevs2)
    ugn4.plot_exact_pdf()
    n4.plot_exact_pdf()
    ugn4.plot_empirical_pdf()
    n4.plot_empirical_pdf()
    ugn4.plot_exact_cdf()
    n4.plot_exact_cdf()
    ugn4.plot_empirical_cdf()
    ugn4.plot_empirical_cdf()


