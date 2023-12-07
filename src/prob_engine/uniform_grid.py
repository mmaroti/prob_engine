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

        # absolute value and normalization is done later
        self._parameter = torch.rand(
            size=torch.Size(self._counts.flatten()),
            dtype=torch.float32,
            device=self.device)

        if False:
            self._parameter = self._parameter.abs()
            self._parameter /= torch.sum(self._parameter)

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
    def cell_size(self) -> torch.Tensor:
        return self._cell_size

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._parameter

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        flat_indices = torch.multinomial(
            self._parameter.view(self._parameter.numel()).abs(),
            sample_shape.numel(),
            replacement=True)

        flat_coords = torch.empty(
            (sample_shape.numel(), self.event_shape.numel()),
            dtype=torch.long, device=self._device)

        for i, d in enumerate(reversed(self._parameter.shape)):
            flat_coords[:, -1 - i] = flat_indices % d
            flat_indices //= d

        coords = flat_coords.view(sample_shape + self.event_shape).float()
        coords += torch.rand(coords.shape,
                             dtype=torch.float32, device=self._device)

        values = self.min_bounds + coords * self.cell_size
        return values

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        sample_shape = sample.shape[:len(sample.shape) - len(self.event_shape)]
        assert sample.shape == sample_shape + self.event_shape

        sample = sample.to(dtype=torch.float32, device=self._device)
        flat_sample = sample.view(torch.Size(
            [sample_shape.numel(), self.event_shape.numel()]))

        flat_coords = ((flat_sample - self.min_bounds) /
                       self._cell_size).floor().long()
        # print(flat_coords)

        flat_indices = torch.empty(flat_sample.shape[0],
                                   dtype=torch.long, device=self._device)
        flat_invalid = torch.zeros(flat_sample.shape[0],
                                   dtype=torch.bool, device=self._device)
        for i, d in enumerate(self._parameter.shape):
            flat_invalid |= flat_coords[:, i] < 0
            flat_invalid |= flat_coords[:, i] >= self._counts[i]
            if i == 0:
                flat_indices = flat_coords[:, 0]
            else:
                flat_indices *= d
                flat_indices += flat_coords[:, i]

        # print(flat_invalid)
        # print(flat_indices)

        flat_indices[flat_invalid] = 0
        # print(flat_indices)

        param = self._parameter.view(self._parameter.numel()).abs()
        param /= torch.sum(param)

        flat_probs = param[flat_indices]
        flat_probs[flat_invalid] = 0.0
        # print(flat_probs)

        return flat_probs.view(sample_shape)


def test():
    if False:
        grid = UniformGrid(torch.tensor(
            [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            torch.tensor([[2, 3], [4, 5]]))
        grid.sample(torch.Size([3, 2]))
        # grid.sample()

    if False:
        grid = UniformGrid(torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0]]), torch.tensor([4, 5]))
        grid.plot_sample_histogram()

    if True:
        grid = UniformGrid(torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0]]), torch.tensor([4, 4]))
        print(grid._parameter)
        print(grid.log_prob(torch.tensor([-0.9, 0.1])))
        print(grid.log_prob(torch.tensor(
            [[-0.9, 0.1], [0.9, -0.1], [-1.1, 0.0]])))
