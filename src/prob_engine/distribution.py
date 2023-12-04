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

from matplotlib import pyplot
import torch
from typing import Iterator


class Distribution:
    def __init__(self, event_shape: torch.Size):
        """
        Creates an abstract multi variate distribution whose events have
        the specified event shape.
        """
        assert isinstance(event_shape, torch.Size)
        self._sample_shape = event_shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of events that this distribution can produce.
        """
        return self._sample_shape

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns the list of parameters of this parametric distribution.
        """
        if False:
            yield

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Randomly samples from the distribution batch many times and returns
        a tensor of shape sample_shape + sample_shape.
        """
        raise NotImplemented()

    def plot_sample_histogram(self):
        numel = self.event_shape.numel()
        if numel == 1:
            samples = self.sample(torch.Size([100000]))
            samples = samples.detach().cpu().flatten().numpy()
            pyplot.hist(samples, bins=60, density=True)
            pyplot.show()
        elif numel == 2:
            samples = self.sample(torch.Size([100000]))
            samples = samples.detach().cpu().reshape((-1, 2)).numpy()
            pyplot.hist2d(samples[:, 0], samples[:, 1],
                          bins=[60, 60], density=True)
            pyplot.show()
        else:
            raise ValueError("invalid sample size")
