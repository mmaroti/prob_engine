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

from typing import Iterator, Optional
import torch
import numpy

from .distribution import Distribution


class Mixture(Distribution):
    def __init__(self,
                 distributions: list[Distribution],
                 weights: torch.Tensor,
                 device: Optional[str] = None):
        assert len(distributions) > 0
        assert len(distributions) == weights.numel()
        assert (weights == 0).all().logical_not()
        assert ( torch.tensor(
                [d.event_shape == distributions[0].event_shape 
                    for d in distributions]
                ) ).all()
    
        Distribution.__init__(self, distributions[0].event_shape, device=device)
        self._distributions = distributions
        self._weights = weights.flatten().abs(
                                ).to(dtype=torch.float32, device=self._device)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def distributions(self) -> list[Distribution]:
        return self._distributions

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._weights
        for d in self._distributions:
            yield from d.parameters

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        counts = numpy.random.multinomial(
            batch_shape.numel(),
            self.weights.flatten().abs()/self.weights.abs().sum() )
        assert counts.sum() == batch_shape.numel()
        samples = torch.cat(
            [d.sample(torch.Size((counts[i], ))) 
             for i, d in enumerate(self.distributions)],
            dim = 0)
        assert samples.shape[0] == batch_shape.numel()
        perm = torch.randperm(batch_shape.numel())
        return samples[perm].view(batch_shape + self.event_shape)


    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[: - len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        norm_weights = self.weights.flatten().abs()/self.weights.abs().sum()
        return torch.stack( 
            [w * d.get_pdf(sample) 
             for w, d in zip(norm_weights, self.distributions)
             ], dim = 0 ).sum(dim = 0)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        return torch.log(self.get_pdf(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        norm_weights = self.weights.flatten().abs()/self.weights.abs().sum()
        return torch.stack( 
            [w * d.get_cdf(sample) 
             for w, d in zip(norm_weights, self.distributions)
             ], dim = 0 ).sum(dim = 0)


def test():
    from .uniform_grid import UniformGrid
    dist = Mixture(
        [
            UniformGrid(torch.tensor(
                [[0.5, 0.5], [1.0, 1.0]]), torch.tensor([1, 1])),
            UniformGrid(torch.tensor(
                [[-1.0, -1.0], [-0.5, -0.5]]), torch.tensor([1, 1])),
            Mixture([
                UniformGrid(torch.tensor(
                    [[-0.5, -0.5], [0.0, 0.0]]), torch.tensor([1, 1])),
                UniformGrid(torch.tensor(
                    [[0.0, 0.0], [0.5, 1.0]]), torch.tensor([1, 1]))
            ], torch.tensor([0.2, 0.8]))
        ], torch.tensor([0.2, 0.3, 0.5])
    )

    print("Event shape", dist.event_shape)
    print("Weights", dist.weights)
    print("Distributions", dist.distributions)
    print("Parameters", list(dist.parameters))
    dist.plot_exact_pdf()
    dist.plot_empirical_pdf()
    dist.plot_empirical_cdf()
    dist.plot_exact_cdf()
