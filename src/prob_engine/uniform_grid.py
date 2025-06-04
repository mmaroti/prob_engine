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

from typing import Callable, Iterator, Optional
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

    def centers(self) -> torch.Tensor:
        """
        Returns all center sample points for all grid cells.
        """
        ranges = [torch.tensor(list(range(0,i))) for i in self.counts.flatten()]
        coords = torch.cartesian_prod( *ranges )
        centers = (self.bounds[0].flatten() 
                   + ( coords + 0.5 ) * self._cell_size.flatten())
        return centers.reshape(self._parameter.shape + self.event_shape)

    def initialize(self, pdf: Callable[[torch.Tensor], float]):
        """
        For each grid cell it calls the given pdf function with
        the center of that grid cell which in turn returns the 
        un-normalized pdf value for that sample.
        """
        centers = self.centers()
        centers = centers.reshape( (self._parameter.numel(), ) + self.event_shape)
        self._parameter = torch.stack(
            [pdf(t) for t in centers],
            -1  ).reshape(self._parameter.shape)

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

        flat_coords = ((flat_sample - self.min_bounds.flatten()) /
                       self._cell_size.flatten()).floor().long()
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

        flat_sample = sample.view( (batch_shape.numel(), self.event_numel) )

        flat_params = self._parameter.flatten().abs()
        flat_params *= 1.0/(flat_params.sum())

        flat_centers = self.centers().view(
             (flat_params.numel(), self.event_numel) )
        cell_lower_corners = flat_centers - self._cell_size.flatten()/2

        excess = (flat_sample.unsqueeze(1) - cell_lower_corners).relu()
        volume = torch.minimum(excess, 
                               self._cell_size.view( 
                                   (1,self.event_numel) 
                                   ) ).prod(-1)
        prob = (volume / self._cell_volume) * flat_params

        return prob.sum(-1).reshape(batch_shape)


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
        grid2.plot_empirical_pdf()
        grid2.plot_exact_pdf()
        grid2.plot_empirical_cdf()
        grid2.plot_exact_cdf()
