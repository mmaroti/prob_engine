# Copyright (C) 2023, Miklos Maroti
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


class Mixture(Distribution):
    def __init__(self,
                 distributions: list[Distribution],
                 weights: torch.Tensor,
                 device: Optional[str] = None):
        """
        Creates a mixture of different distributions, that is,
        the CDF of the mixture is a linear combination of different individual CDFs.
        """
        assert (
            weights.numel() > 1 and
            len(distributions) == weights.numel() and
            torch.abs(weights).sum() > 0 and
            torch.all(torch.tensor(
                [d.event_shape == distributions[0].event_shape for d in distributions]))
        ), "Shape mismatch or weight error detected in Mixture distribution."

        # Check whether provided distributions contain Mixtures. If they do, "unpack" them.
        subDistributions = [d._distributions if isinstance(
            d, Mixture) else [d] for d in distributions]
        correct_weights = torch.abs(weights)/torch.abs(weights).sum()
        subweights = [w * d.weights if isinstance(d, Mixture) else w.unsqueeze(-1)
                      for w, d in zip(correct_weights, distributions)]

        flat_weights = torch.cat(tensors=subweights)
        flat_distributions: list[Distribution] = [x
                                                  for xs in subDistributions
                                                  for x in xs]

        assert flat_weights.numel() == len(
            flat_distributions), "Number of modified weights and distributions does not match!"
        assert torch.all(torch.tensor([d.event_shape == flat_distributions[0].event_shape for d in flat_distributions])
                         ), "Unpacked distributions don't have matching event_shape values!"

        Distribution.__init__(
            self, flat_distributions[0].event_shape, device=device)
        self._distributions = flat_distributions
        self._weights = flat_weights.to(
            dtype=torch.float32, device=self._device)

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
        # TODO: make it more efficient, do not sample unnecessarily
        samples = [d.sample(batch_shape) for d in self.distributions]
        selections = torch.multinomial(
            self._weights.flatten().abs(),
            batch_shape.numel(),
            replacement=True)
        selected = [(selections == torch.tensor(float(i))).unsqueeze(-1) * samples[i].flatten(end_dim=-len(self.event_shape)-1)
                    for i in list(range(self._weights.numel()))
                    ]
        return sum(selected).reshape(batch_shape + self.event_shape)

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        return sum([w * d.get_pdf(sample) for w, d in zip(self.weights, self.distributions)])

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log(self.get_pdf(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        return sum([w * d.get_cdf(sample) for w, d in zip(self.weights, self.distributions)])


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
        ],
        torch.tensor([0.05, 0.2, 0.5])
    )

    print("Event shape", dist.event_shape)
    print("Weights", dist.weights)
    print("Distributions", dist.distributions)
    print("Parameters", list(dist.parameters))
    dist.plot_exact_pdf()
    dist.plot_empirical_pdf()
    dist.plot_empirical_cdf()
    dist.plot_exact_cdf()
