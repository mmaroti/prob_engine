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
                 device: Optional[str] = None):
        assert len(distributions) > 0
        assert all(d.event_shape == distributions[0].event_shape
                   for d in distributions)
        assert all(d._device == distributions[0]._device
                   for d in distributions)

        Distribution.__init__(
            self, distributions[0].event_shape, device=device)
        assert distributions[0]._device == self._device
        self._distributions = distributions
        self._weights = torch.nn.Parameter(torch.rand(
            size=[len(distributions)],
            dtype=torch.float32,
            device=self._device))

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
        norm_weights = self._weights.abs()
        norm_weights /= norm_weights.sum()
        counts = numpy.random.multinomial(
            n=batch_shape.numel(),
            pvals=norm_weights.detach().numpy())
        assert counts.sum() == batch_shape.numel()
        samples = torch.cat(
            [d.sample(torch.Size((c, )))
             for c, d in zip(counts, self._distributions)],
            dim=0)
        samples = samples.to(device=self._device)
        assert samples.shape[0] == batch_shape.numel()
        perm = torch.randperm(batch_shape.numel(), device=self._device)
        return samples[perm].view(batch_shape + self._event_shape)

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[: - len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.to(device=self._device)

        norm_weights = self._weights.abs()
        norm_weights /= norm_weights.sum()
        result = torch.stack([d.get_pdf(sample)
                             for d in self._distributions], dim=-1)
        return (result * norm_weights).sum(-1)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log(self.get_pdf(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[: - len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.to(device=self._device)

        norm_weights = self._weights.abs()
        norm_weights /= norm_weights.sum()
        result = torch.stack([d.get_cdf(sample)
                              for d in self._distributions], dim=-1)
        return (result * norm_weights).sum(-1)


def test():
    from .uniform_grid import UniformGrid
    from .uniform_ball import UniformBall
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
            ]),
            UniformBall(torch.tensor([-0.5, 0.5]), torch.tensor(0.4)),
            UniformBall(torch.tensor([0.35, -0.5]), torch.tensor(0.25)),
            UniformBall(torch.tensor([0.65, -0.5]), torch.tensor(0.25))
        ]
    )

    print("Event shape", dist.event_shape)
    print("Weights", dist.weights)
    print("Distributions", dist.distributions)
    print("Parameters", list(dist.parameters))
    dist.plot_exact_pdf()
    dist.plot_empirical_pdf()
    dist.plot_empirical_cdf()
    dist.plot_exact_cdf()
