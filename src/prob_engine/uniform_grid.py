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

from typing import Iterator
import torch

from .distribution import Distribution


class UniformGrid(Distribution):
    def __init__(self,
                 bounds: torch.Tensor,
                 counts: torch.Tensor):
        assert bounds.shape[0] == 2 and bounds.shape[1:] == counts.shape
        assert torch.all(bounds[0] < bounds[1]).item() \
            and torch.all(counts >= 1).item()

        Distribution.__init__(self, counts.shape)
        self._bounds = bounds.detach().float()
        self._counts = counts.detach().cpu().int()

        value = torch.rand(
            size=torch.Size(self._counts.flatten()),
            dtype=self._bounds.dtype,
            device=self._bounds.device)
        self._parameter = value / torch.sum(value)

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
    def cell_size(self) -> torch.Tensor:
        return (self._bounds[1] - self._bounds[0]) / self._counts

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._parameter

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        print("batch shape:", batch_shape)
        print("sample shape:", self.sample_shape)
        print("parameter shape:", self._parameter.shape)

        flat_indices = torch.multinomial(
            self._parameter.flatten(),
            batch_shape.numel(),
            replacement=True)
        print("flat indices shape:", flat_indices.shape)
        print("flat indices:", flat_indices)

        flat_coords = torch.empty(
            (flat_indices.shape[0], self.sample_shape.numel()),
            dtype=torch.long)
        print("flat coords shape:", flat_coords.shape)

        for i, d in enumerate(reversed(self._parameter.shape)):
            flat_coords[:, -1 - i] = flat_indices % d
            flat_indices //= d
        print("flat coords:")
        print(flat_coords)

        coords = flat_coords.reshape(batch_shape + self.sample_shape)
        print("coords shape:", coords.shape)
        print("coords:")
        print(coords)

        coords = coords.float() + torch.rand(coords.shape)

        print("min_bounds:", self.min_bounds)
        print("max_bounds:", self.max_bounds)
        print("cell size:", self.cell_size)

        values = self.min_bounds + coords * self.cell_size
        print("values:", values)


def test():
    grid = UniformGrid(torch.tensor(
        [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        torch.tensor([[2, 3], [4, 5]]))
    grid.sample(torch.Size([3, 2]))
    # grid.sample()
