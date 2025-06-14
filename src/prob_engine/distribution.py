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
        self._event_shape = event_shape

        # device = torch.device("cpu")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            assert isinstance(device, torch.device)
        self._device = device

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of events that this distribution can produce.
        """
        return self._event_shape

    @property
    def event_numel(self) -> int:
        """
        Returns the dimension of the distribution (or the number of element
        in a event).
        """
        return self._event_shape.numel()

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

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Randomly samples from the distribution possibly multiple times and
        returns a tensor of shape batch_shape + event_shape.
        """
        raise NotImplementedError()

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logarithm of the probability density function at the
        given sample. The input is of shape batch_shape + event_shape and
        the output is of shape batch_shape.
        """
        raise NotImplementedError()

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the density function at the given sample.
        The input is of shape batch_shape + event_shape and
        the output is of shape batch_shape.
        """
        return torch.exp(self.log_prob(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cumulative distribution function at the given sample.
        The input is of shape batch_shape + event_shape and the output is of
        shape batch_shape.
        """
        raise NotImplementedError()

    def get_rectangle_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the exact probability measure of an [a,b) rectangle
        using the CDF function of the distribution.
        For each rectangle, evaluates CDF at the 2^(self.event_numel)
        corners of the rectangle, which then yields the probability.
        """
        batch_shape = sample.shape[:-len(self._event_shape) - 1]
        assert sample.shape == batch_shape + (2,) + self._event_shape
        sample = sample.view((batch_shape.numel(), 2, self.event_numel)
                             ).to(device=self._device)
        sides = sample[:, 1, :] - sample[:, 0, :]
        assert (sides > 0).all()
        combs01 = torch.bitwise_and(
            torch.arange(2**self.event_numel,
                         device=self._device).unsqueeze(-1),
            2**torch.arange(self.event_numel, device=self._device))
        combs01 = (combs01 > 0).view((2**self.event_numel, self.event_numel))
        corners = sample[:, 0, :].view(
            (batch_shape.numel(), 1, self.event_numel))\
            + sides.view((batch_shape.numel(), 1, self.event_numel)) \
            * combs01
        result = (self.get_cdf(corners) *
                  (2*(combs01.count_nonzero(-1) % 2)-1)
                  ).sum(-1)
        return result.view(batch_shape)

    def get_empirical_cdf(self,
                          count: int,
                          sample: torch.Tensor) -> torch.Tensor:
        """
        Generates 'count' many random samples, and uses them to evaluate the
        empirical distribution function on 'sample'. The input 'sample' is of
        shape batch_shape + event_shape and the output is of shape batch_shape.
        """

        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view((batch_shape.numel(), self.event_numel)
                             ).to(device=self._device)
        points = self.sample(torch.Size((count, ))
                             ).view(torch.Size((count, 1, self.event_numel))
                                    ).to(device=self._device)
        result = (points <= sample).all(-1).count_nonzero(0) / count
        return result.view(batch_shape)

    def get_empirical_prob_rectangle(self, count: int,
                                     rectangles: torch.Tensor) -> torch.Tensor:
        """
        Generates 'count' many random samples, and uses them to approximate
        the probability of a randomly sample from the distribution
        falling into given [a,b) hyper rectangle.
        """
        batch_shape = rectangles.shape[:-len(self._event_shape)-1]
        assert rectangles.shape == batch_shape + (2,) + self._event_shape
        upper_bounds = rectangles.view((batch_shape.numel(), 2, 1,
                                        self.event_numel)
                                       )[:, 1, :, :].to(device=self._device)
        lower_bounds = rectangles.view((batch_shape.numel(), 2, 1,
                                        self.event_numel)
                                       )[:, 0, :, :].to(device=self._device)
        points = self.sample(torch.Size((count, ))
                             ).view(torch.Size((1, count, self.event_numel))
                                    ).to(device=self._device)
        result = torch.logical_and(
            points >= lower_bounds,
            points < upper_bounds
        ).all(-1).count_nonzero(-1) / count
        return result.view(batch_shape)

    def get_empirical_prob_ball(self, count: int,
                                balls: torch.Tensor) -> torch.Tensor:
        """
        Generates 'count' many random samples, and uses them to approximate
        the probability of a randomly sample from the distribution
        falling into given (open) hyper balls.
        If self.event_numel=k, balls should be given in [x_1,...,x_k,r] form, where
        [x_1,...,x_k] correspond to the center, and r corresponds to the radius.
        """
        batch_shape = balls.shape[:-1]
        assert balls.shape == batch_shape + (self.event_numel + 1, )
        balls = balls.view((batch_shape.numel(), self.event_numel + 1))
        radius = balls[:, -1].view((batch_shape.numel(), 1))
        center = balls[:, :-1]
        center = center.view((batch_shape.numel(), 1,
                              self.event_numel)
                             ).to(device=self._device)
        points = self.sample(torch.Size((count, ))
                             ).view(torch.Size((1, count, self.event_numel))
                                    ).to(device=self._device)
        distance = (points - center).pow(2).sum(-1)
        result = (distance < radius.pow(2)).count_nonzero(-1)/count
        return result.view(batch_shape)

    def plot_empirical_pdf(self,
                           min_bound: float = -1.0,
                           max_bound: float = 1.0,
                           bins: int = 60,
                           count: int = 100000):
        """
        Takes count many samples from the distribution and plots the resulting
        histogram approximating the probability density of the distribution.
        This method assumes that the dimension of the distribution is one or two.
        """
        if self.event_numel == 1:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().flatten().detach().numpy()
            pyplot.hist(sample,
                        bins=bins,
                        range=(min_bound, max_bound),
                        density=True)
            pyplot.show()
        elif self.event_numel == 2:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().reshape((count, 2)).detach().numpy()
            pyplot.hist2d(sample[:, 0], sample[:, 1],
                          bins=bins,
                          range=((min_bound, max_bound),
                                 (min_bound, max_bound)),
                          density=True,
                          rasterized=True)
            pyplot.colorbar()
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_empirical_cdf(self,
                           min_bound: float = -1.0,
                           max_bound: float = 1.0,
                           bins: int = 120,
                           count: int = 100000):
        """
        Takes count many samples from the distribution and plots the resulting
        cumulative histogram approximating the cumulative distribution function.
        This method assumes that the dimension of the distribution is one.
        """
        if self.event_numel == 1:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().flatten().detach().numpy()
            pyplot.hist(sample,
                        bins=bins,
                        range=(min_bound, max_bound),
                        density=True,
                        cumulative=True)
            pyplot.show()
        elif self.event_numel == 2:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().reshape((count, 2)).detach().numpy()
            values, xs, ys = numpy.histogram2d(
                sample[:, 0], sample[:, 1],
                bins=bins,
                range=((min_bound, max_bound),
                       (min_bound, max_bound)))
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

    def plot_exact_pdf(self,
                       min_bound: float = -1.0,
                       max_bound: float = 1.0,
                       bins: int = 60):
        """
        Creates a grid of sample points and plots the corresponding probability
        density values as calculated by the log_prob method. This method assumes
        that the dimension of the distribution is one or two.
        """
        if self.event_numel == 1:
            width = (max_bound - min_bound) / bins
            sample = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample = sample.view(torch.Size((bins,)) + self._event_shape)
            value = self.get_pdf(sample)
            pyplot.bar(
                x=sample.cpu().flatten().numpy(),
                height=value.cpu().flatten().detach().numpy(),
                width=width)
            pyplot.show()
        elif self.event_numel == 2:
            width = (max_bound - min_bound) / bins
            sample1 = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
            sample2 = torch.stack(sample2, dim=-1).view(
                torch.Size((bins, bins)) + self._event_shape)
            value2 = self.get_pdf(sample2)
            pyplot.pcolormesh(
                sample1.cpu().numpy(),
                sample1.cpu().numpy(),
                value2.cpu().detach().numpy(),
                rasterized=True)
            pyplot.colorbar()
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_exact_cdf(self,
                       min_bound: float = -1.0,
                       max_bound: float = 1.0,
                       bins: int = 120):
        """
        Creates a grid of sample points and plots the corresponding cumulative
        distribution function values as calculated by the get_cdf method. This
        method assumes that the dimension of the distribution is one or two.
        """
        if self.event_numel == 1:
            width = (max_bound - min_bound) / bins
            sample = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample = sample.view(torch.Size((bins,)) + self._event_shape)
            value = self.get_cdf(sample).detach()
            pyplot.bar(
                x=sample.cpu().flatten().numpy(),
                height=value.cpu().flatten().numpy(),
                width=width)
            pyplot.show()
        elif self.event_numel == 2:
            width = (max_bound - min_bound) / bins
            sample1 = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
            sample2 = torch.stack(sample2, dim=-1).view(
                torch.Size((bins, bins)) + self._event_shape)
            value2 = self.get_cdf(sample2).detach()
            pyplot.pcolormesh(
                sample1.cpu().numpy(),
                sample1.cpu().numpy(),
                value2.cpu().numpy(),
                rasterized=True)
            pyplot.colorbar()
            pyplot.show()
        else:
            raise ValueError("invalid event size")
