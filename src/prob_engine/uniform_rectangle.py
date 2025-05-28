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

from .distribution import Distribution

class UniformRectangle(Distribution):
    def __init__(self,
                 ab: torch.Tensor,
                 device: Optional[str] = None):
        assert ab.shape[0] == 2
        assert ab.dim() > 1
        assert (ab[0] <= ab[-1]).all()
        Distribution.__init__(self, ab[0].shape, device=device)
        self._ab = ab.to(dtype=torch.float32, device=self._device)

    @property
    def ab(self) -> torch.Tensor:
        return self._ab

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._ab

    def measure(self) -> torch.Tensor:
        assert (self.ab[0] <= self.ab[-1]).all()
        return (self.ab[-1].flatten() - self.ab[0].flatten()).prod()
    
    def sample(self, batch_shape: torch.Size) -> torch.Tensor:
        return ( torch.rand(batch_shape + self.event_shape) 
                * (self.ab[-1] - self.ab[0]) + self.ab[0] )

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self.ab[0] < self.ab[-1]).all(), "No density function exists!"
        batch_shape = sample.shape[:-len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        inside = ( ( self.ab[0].flatten()
                    <= sample.view( batch_shape + (self.event_numel, ) ) ) 
                  * ( self.ab[-1].flatten()
                     >= sample.view( batch_shape + (self.event_numel, ) ) )
                         )
        return inside.all(-1)/self.measure()
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log( self.get_pdf(sample) )

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        if (self.ab[0] < self.ab[-1]).all():
            diffs = torch.maximum(
                    torch.minimum(
                            sample.view( batch_shape + (self.event_numel, ) ),
                            self.ab[-1].flatten()
                            ) - self.ab[0].flatten(),
                    torch.zeros(self.event_numel)
                    )
            return diffs.prod(-1).view(batch_shape) / self.measure()
        else:
            """ Even if the rectangle is lower dimensional,
            i.e., a[i]=b[i] for some i, the CDF can be defined."""
            raise NotImplemented()

def test():
    dist1 = UniformRectangle( torch.tensor([[-0.5], [0.5]]) )
    print("Parameters", list(dist1.parameters))
    print("Measure", dist1.measure())
    dist1.plot_empirical_pdf()
    dist1.plot_exact_pdf()
    dist1.plot_empirical_cdf()
    dist1.plot_exact_cdf()

    dist2 = UniformRectangle( torch.tensor([[-0.5,-0.5], [0.0,0.0]]) )
    print("Parameters", list(dist2.parameters))
    print("Measure", dist2.measure())
    dist2.plot_empirical_pdf()
    dist2.plot_exact_pdf()
    dist2.plot_empirical_cdf()
    dist2.plot_exact_cdf()