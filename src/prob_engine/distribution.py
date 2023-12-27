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

import numpy
from matplotlib import pyplot
import torch
from typing import Iterator, Optional


class Distribution:
    def __init__(self, event_shape: torch.Size, device: Optional[str] = None):
        """
        Creates a multi variate distribution whose events have the specified
        event shape on the given device. If the device is not specified, then
        it will be automatically selected between cuda and cpu.
        """
        assert isinstance(event_shape, torch.Size)
        self._sample_shape = event_shape

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of events that this distribution can produce.
        """
        return self._sample_shape

    @property
    def device(self) -> str:
        """
        Returns the underlying device (cuda or cpu) for the tensors this
        distribution can work with.
        """
        return self._device

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
        a tensor of shape sample_shape + event_shape.
        """
        raise NotImplemented()

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logarithm of the probability density function at the
        given sample. The input is of shape sample_shape + event_shape and
        the output is of shape sample_shape.
        """
        raise NotImplemented()

    def plot_sample_density(self,
                            bins: int = 60,
                            count: int = 100000):
        """
        Takes count many samples from the distribution and plots the resulting
        histogram approximating the probability density of the distribution.
        This method assumes that the dimension of the distribution is one or two.
        """
        numel = self.event_shape.numel()
        if numel == 1:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().flatten().numpy()
            pyplot.hist(sample, bins=bins, density=True)
            pyplot.show()
        elif numel == 2:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().reshape((-1, 2)).numpy()
            pyplot.hist2d(sample[:, 0], sample[:, 1],
                          bins=bins, density=True,
                          rasterized=True)
            pyplot.colorbar()
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_sample_cumulative(self,
                               bins: int = 120,
                               count: int = 100000):
        """
        Takes count many samples from the distribution and plots the resulting
        cumulative histogram approximating the cumulative distribution function.
        This method assumes that the dimension of the distribution is one.
        """
        numel = self.event_shape.numel()
        if numel == 1:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().flatten().numpy()
            pyplot.hist(sample, bins=bins, density=True, cumulative=True)
            pyplot.show()
        elif numel == 2:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().reshape((-1, 2)).numpy()
            values, xs, ys = numpy.histogram2d(
                sample[:, 0], sample[:, 1], bins=bins)
            values = numpy.float32(values)
            values *= 1.0 / count
            values = values.cumsum(axis=0).cumsum(axis=1)
            xs, ys = numpy.meshgrid(xs, ys)
            pyplot.pcolormesh(
                xs, ys,
                numpy.transpose(values),
                rasterized=True)
            pyplot.colorbar()
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_exact_density(self,
                           min_bound: float = -1.0,
                           max_bound: float = 1.0,
                           bins: int = 60):
        """
        Creates a grid of sample points and plots the corresponding probability
        density values as calculated by the log_prob method. This method assumes
        that the dimension of the distribution is one or two.
        """
        numel = self.event_shape.numel()
        if numel == 1:
            width = (max_bound - min_bound) / bins
            sample = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample = sample.view(torch.Size((bins,)) + self._sample_shape)
            prob = torch.exp(self.log_prob(sample))
            pyplot.bar(
                x=sample.cpu().flatten().numpy(),
                height=prob.cpu().flatten().numpy(),
                width=width)
            pyplot.show()
        elif numel == 2:
            width = (max_bound - min_bound) / bins
            sample1 = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample2 = torch.meshgrid((sample1, sample1), indexing="xy")
            sample2 = torch.stack(sample2, dim=-1).view(
                torch.Size((bins, bins)) + self._sample_shape)
            prob2 = torch.exp(self.log_prob(sample2))
            pyplot.pcolormesh(
                sample1.cpu().numpy(),
                sample1.cpu().numpy(),
                prob2.cpu().numpy(),
                rasterized=True)
            pyplot.colorbar()
            pyplot.show()
        else:
            raise ValueError("invalid event size")
