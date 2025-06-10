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
        Returns center points for all grid cells.
        """
        ranges = [torch.arange(start=0, end = i,
                               device = self._device)
                               for i in self.counts.flatten()]
        coords = torch.cartesian_prod( *ranges )
        centers = (self.bounds[0].flatten() 
                   + ( coords + 0.5 ) * self._cell_size.flatten())
        return centers.reshape(self._parameter.shape + self.event_shape)
    
    def cell_bounds(self) -> torch.Tensor:
        """
        Returns pairs of lower and upper bounds for all grid cells.
        """
        event_dims = len(self._event_shape)
        centers = self.centers().unsqueeze(-event_dims-1)
        half_size = self._cell_size.unsqueeze(-event_dims-1)/2
        cell_bounds = torch.cat(
            [centers - half_size, centers + half_size],
            dim = -event_dims-1)
        assert cell_bounds.shape == self._parameter.shape \
                                    + (2,) + self._event_shape
        return cell_bounds

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
    
    def initialize_from_distribution_pdf_center(self, target: Distribution):
        """
        Evaluates get_pdf function of target distribution then
        sets parameters to approximate the result.
        """
        assert target._event_shape == self._event_shape
        centers = self.centers().to(device = target._device)
        pdf = target.get_pdf(centers).to(device = self._device)
        self._parameter = pdf.view(self._parameter.shape)

    def initialize_from_distribution_pdf_rectangle(self, target: Distribution):
        """
        Evaluates get_pdf function of target distribution then
        sets parameters to approximate the result.
        """
        assert target._event_shape == self._event_shape
        bounds = self.cell_bounds().to(device = target._device)
        probs = target.get_rectangle_prob(bounds).to(device = self._device)
        self._parameter = probs.view(self._parameter.shape)
        
    def initialize_from_sample(self, sample: torch.Tensor):
        """
        Uses provided sample to approximate probabilities
        of underlying distribution falling into [a,b) grid cells,
        then sets parameters based on results.
        """
        batch_shape = sample.shape[:len(sample.shape) - len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape.numel() + (self.event_numel,)
                             ).to(dtype=torch.float32, device=self._device)
        centers = self.centers().view(self._parameter.shape + (self.event_numel))
        half_cell_size = self._cell_size.flatten()/2
        cell_lower_bounds = ( centers - half_cell_size ).unsqueeze(-2)
        cell_upper_bounds = ( centers + half_cell_size ).unsqueeze(-2)
        cell_probs = torch.logical_and(
                                    cell_lower_bounds < sample,
                                    cell_upper_bounds >= sample
                                    ).all(-1).count_nonzero(-1) / batch_shape.numel()
        self._parameter = cell_probs.view(self._parameter.shape)

    def initialize_from_distribution_empirical(self, count: int, target: Distribution):
        """
        Generates 'count' many samples from 'target' distribution,
        approximate probabilities of samples falling within grid cells,
        and uses the results to set the parameters.
        """
        assert self._event_shape == target._event_shape
        bounds = self.cell_bounds().to(device = target._device)
        probs = target.get_empirical_prob_rectangle(count, bounds)
        self._parameter = probs.to(device = self.device).view(self._parameter.shape)

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

        coords = flat_coords.view(batch_shape + self._event_shape).float()
        coords += torch.rand(coords.shape,
                             dtype=torch.float32, device=self._device)

        values = self.min_bounds + coords * self._cell_size
        return values

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:len(sample.shape) - len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape

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
        return torch.log(self.get_pdf(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:len(sample.shape) - len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.to(dtype=torch.float32, device=self._device)
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
        return prob.sum(-1).view(batch_shape)


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
