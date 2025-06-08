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
                 bounds: torch.Tensor,
                 device: Optional[str] = None):
        assert bounds.shape[0] == 2
        assert bounds.dim() > 1
        assert (bounds[0] <= bounds[-1]).all()
        Distribution.__init__(self, bounds[0].shape, device=device)
        self._bounds = bounds.to(dtype=torch.float32, device=self._device)

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        return self._bounds[0]

    @property
    def max_bounds(self) -> torch.Tensor:
        return self._bounds[-1]
    
    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._bounds

    def measure(self) -> torch.Tensor:
        return (self.max_bounds-self.min_bounds).relu().prod()
    
    def sample(self, batch_shape: torch.Size) -> torch.Tensor:
        rands = torch.rand(batch_shape + self._event_shape, device = self._device) 
        return rands * (self.max_bounds - self.min_bounds) + self.min_bounds

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self.min_bounds < self.max_bounds).all(), "No density function exists!"
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (self.event_numel, )
                            ).to(device = self._device)
        inside = torch.logical_and(
                (self.min_bounds.flatten() <= sample),
                (self.max_bounds.flatten() >= sample))
        return inside.all(-1)/self.measure()
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log( self.get_pdf(sample) )

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (self.event_numel, )
                            ).to(device = self._device)

        if (self.min_bounds < self.max_bounds).all():
            diffs = (torch.minimum( sample, self.max_bounds.flatten() )
                     - self.min_bounds.flatten() ).relu()
                    
            return diffs.prod(-1).view(batch_shape) / self.measure()
        else:
            """ Even if the rectangle is lower dimensional,
            i.e., a[i]=b[i] for some i, the CDF can be defined."""
            raise NotImplementedError()

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