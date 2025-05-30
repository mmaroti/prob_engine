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

from .distribution import Distribution


class UniformGrid(Distribution):
    def __init__(self,
                 bounds: torch.Tensor,
                 counts: torch.Tensor,
                 device: Optional[str] = None):
        assert bounds.shape[0] == 2 and bounds.shape[1:] == counts.shape
        assert torch.all(bounds[0] < bounds[1]).item() \
            and torch.all(counts >= 1).item()

        Distribution.__init__(self, counts.shape, device=device)
        self._bounds = bounds.to(dtype=torch.float32, device=self._device)
        self._counts = counts.to(dtype=torch.int32, device=self._device)
        self._cell_size = (self._bounds[1] - self._bounds[0]) / self._counts
        self._cell_volume = self._cell_size.flatten().prod()

        # absolute value and normalization is done later
        self._parameter = torch.nn.Parameter(torch.rand(
            size=torch.Size(self._counts.flatten().tolist()),
            dtype=torch.float32,
            device=self._device))

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        return self._bounds[0]

    @property
    def max_bounds(self) -> torch.Tensor:
        return self._bounds[1]

    @property
    def counts(self) -> torch.Tensor:
        return self._counts

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._parameter

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        flat_indices = torch.multinomial(
            self._parameter.flatten().abs(),
            batch_shape.numel(),
            replacement=True)

        flat_coords = torch.empty(
            (batch_shape.numel(), self.event_numel),
            dtype=torch.long, device=self._device)

        for i, d in enumerate(reversed(self._parameter.shape)):
            flat_coords[:, -1 - i] = flat_indices % d
            flat_indices //= d

        coords = flat_coords.view(batch_shape + self.event_shape).float()
        coords += torch.rand(coords.shape,
                             dtype=torch.float32, device=self._device)

        values = self.min_bounds + coords * self._cell_size
        return values

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        sample = sample.to(dtype=torch.float32, device=self._device)
        flat_sample = sample.view(torch.Size(
            [batch_shape.numel(), self.event_numel]))

        flat_coords = ((flat_sample - self.min_bounds) /
                       self._cell_size).floor().long()
        flat_counts = self._counts.flatten()

        flat_indices = torch.empty(flat_sample.shape[0],
                                   dtype=torch.long,
                                   device=self._device)
        flat_invalid = torch.zeros(flat_sample.shape[0],
                                   dtype=torch.bool,
                                   device=self._device)
        for i, d in enumerate(self._parameter.shape):
            flat_invalid |= flat_coords[:, i] < 0
            flat_invalid |= flat_coords[:, i] >= flat_counts[i]
            if i == 0:
                flat_indices = flat_coords[:, 0]
            else:
                flat_indices *= d
                flat_indices += flat_coords[:, i]

        flat_indices[flat_invalid] = 0

        flat_param = self._parameter.flatten().abs()
        flat_param *= 1.0 / (self._cell_volume * torch.sum(flat_param))

        flat_probs = flat_param[flat_indices]
        flat_probs[flat_invalid] = 0.0

        return flat_probs.view(batch_shape)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        return torch.log(self.get_pdf(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        # TODO: Finish implementation
        batch_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape

        if self.event_numel == 1:
            flat_sample = sample.view(
                torch.Size((batch_shape.numel(), self.event_numel)))
            flat_params = self._parameter.flatten().abs()
            flat_params *= 1.0/(flat_params.sum())
            indexes = torch.tensor(list(range(1, flat_params.numel()+1)))
            flat_coords = ((flat_sample - self.min_bounds) / self._cell_size
                           ).floor().relu().long()
            full_cover = ((indexes <= flat_coords)
                          * flat_params).sum(-1)
            partial_cover = (flat_sample -
                             (self.min_bounds
                              + flat_coords * self._cell_size.flatten())
                             ).relu().flatten() * ((indexes - 1 == flat_coords)
                                                   * flat_params).sum(-1) / self._cell_volume
            return (full_cover + partial_cover).view(batch_shape)
        elif self.event_numel == 2:
            raise NotImplementedError()
        else:
            raise NotImplementedError()


def test():
    if False:
        grid = UniformGrid(torch.tensor(
            [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            torch.tensor([[2, 3], [4, 5]]))
        print(grid.sample(torch.Size([3, 2])))
        print(grid.sample())

    if False:
        grid = UniformGrid(torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0]]), torch.tensor([4, 4]))
        print(grid._parameter)
        print(grid.log_prob(torch.tensor([-0.9, 0.1])))
        print(grid.log_prob(torch.tensor(
            [[-0.9, 0.1], [0.9, -0.1], [-1.1, 0.0]])))

    if True:
        grid1 = UniformGrid(
            torch.tensor([[-0.75], [0.75]]),
            torch.tensor([4]))
        grid1.plot_empirical_pdf()
        grid1.plot_exact_pdf()
        grid1.plot_empirical_cdf()
        grid1.plot_exact_cdf()

        grid2 = UniformGrid(
            torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
            torch.tensor([2, 2]))
        print(grid2.event_shape)
        # grid = UniformGrid(torch.tensor([-1, 1]), torch.tensor(5))
        grid2.plot_exact_pdf()
        # grid.plot_exact_cdf()
        grid2.plot_empirical_pdf()
        grid2.plot_empirical_cdf()
