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
    def __init__(self, event_size: int, device: Optional[str] = None):
        """
        Creates a multi variate distribution whose events have the specified
        event shape on the given device. If the device is not specified, then
        it will be automatically selected between cuda and cpu.
        """
        assert isinstance(event_size, int)
        self._event_size = event_size

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
        return torch.Size([self._event_size])

    @property
    def event_size(self) -> int:
        """
        Returns the dimension of the distribution (or the number of element
        in a event).
        """
        return self._event_size

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
        raise NotImplementedError()

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
        return torch.log(self.get_pdf(sample))

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the density function at the given sample.
        The input is of shape batch_shape + event_shape and
        the output is of shape batch_shape.
        If the cdf function exists,
        uses its mixed partial derivative at sample points.
        """
        from . import derivatives
        if sample.requires_grad == False:
            sample.requires_grad_(True)
        coords = list(range(0, self._event_size))
        batch_shape = sample.shape[:-1]
        assert sample.shape[-1] == self.event_shape
        flat_sample = sample.view(
            batch_shape.numel(), self.event_size)
        cdf = self.get_cdf(flat_sample)
        if cdf.numel() == 1:
            result = derivatives.get_partial(
                flat_sample, cdf, coords)
        else:
            result = derivatives.get_partial_batched(
                flat_sample, cdf, coords)

        return result.view(batch_shape)

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cumulative distribution function at the given sample.
        The input is of shape batch_shape + event_shape and the output is of
        shape batch_shape.
        """
        raise NotImplementedError()

    def get_cdf_marginal(self, coords: torch.Tensor,
                         sample: torch.Tensor) -> torch.Tensor:
        """
        Returns cdf of the marginal distribution, where 'coords'
        contains boolean values, and dimensions corresponding to
        'True' are kept, and ones corresponding to 'False' are not.
        In 'coords', values of 0 are taken to be 'False',
        values other than 0 are taken to be 'True'.
        """
        assert coords.shape == self.event_shape
        margin_dim = int(coords.count_nonzero().item())
        batch_shape = sample.shape[:-1]
        assert sample.shape[-1] == margin_dim
        sample = sample.view((batch_shape.numel(), margin_dim))
        points = torch.empty((batch_shape.numel(), self._event_size))
        points[:, torch.logical_not(coords)] = \
            torch.fill(points[:, torch.logical_not(coords)], torch.inf)
        points[:, coords > 0] = sample
        return self.get_cdf(points).view(batch_shape)
    
    def get_pdf_marginal(self, coords: torch.Tensor,
                         sample: torch.Tensor) -> torch.Tensor:
        """
        Returns pdf of the marginal distribution, where 'coords'
        contains boolean values, and dimensions corresponding to
        'True' are kept, and ones corresponding to 'False' are not.
        In 'coords', values of 0 are taken to be 'False',
        values other than 0 are taken to be 'True'.
        e.g. For (X,Y), if coords=[1,0], returns the pdf of X.
        """
        from . import derivatives
        assert coords.shape == self.event_shape
        margin_dim = int(coords.count_nonzero().item())
        batch_shape = sample.shape[:-1]
        assert sample.shape[-1] == margin_dim
        sample = sample.view((batch_shape.numel(), margin_dim))
        if sample.requires_grad == False:
            sample.requires_grad_(True)
        cdf = self.get_cdf_marginal(coords, sample)
        mix_coords = list(range(0, margin_dim))
        if cdf.numel() == 1:
            result = derivatives.get_partial(
                sample, cdf, mix_coords)
        else:
            result = derivatives.get_partial_batched(
                sample, cdf, mix_coords)
        return result
    
    def get_pdf_conditional(self, coords: torch.Tensor,
                         sample: torch.Tensor) -> torch.Tensor:
        """
        Returns conditional pdf, where 'coords' denotes
        which coordinates are taken to be conditional.
        For (X,Y), if coords=[1,0], returns
        f_(X,Y)(x,y)/f_Y(y) if f_Y(y)>0, otherwise f_X(x),
        where 'f' denotes probability density functions
        (the choice of f_X(x) here is arbitrary).
        Sometimes denoted by f_{X|Y}(x|y).
        """
        assert coords.shape == self.event_shape
        margin_dim = int(coords.count_nonzero().item())
        coords = (coords.abs() > 0)
        batch_shape = sample.shape[:-1]
        assert sample.shape[-1:] == self.event_shape
        sample = sample.view((batch_shape.numel(), self.event_size))
        if sample.requires_grad == False:
            sample.requires_grad_(True)
        f_XY = self.get_pdf(sample)
        f_X = self.get_pdf_marginal(
            coords, sample[:, coords > 0])
        f_Y = self.get_pdf_marginal(
            torch.logical_not(coords),
            sample[:, torch.logical_not(coords)])
        result = torch.empty(batch_shape)
        fy_nz = f_Y > 0
        fy_z = torch.logical_not(fy_nz)
        result[fy_nz] = f_XY[fy_nz]/f_Y[fy_nz]
        result[fy_z] = f_X[fy_z]
        return result

    def get_rectangle_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the exact probability measure of an [a,b] rectangle
        using the CDF function of the distribution.
        For each rectangle, evaluates CDF at the 2^(self.event_numel)
        corners of the rectangle, which then yields its measure
        by way of the inclusion-exclusion principle.
        """
        batch_shape = sample.shape[:-2]
        assert sample.shape == batch_shape + (2, self._event_size)
        sample = sample.view((batch_shape.numel(), 2, self.event_size)
                             ).to(device=self._device)
        sides = sample[:, 1, :] - sample[:, 0, :]
        assert (sides > 0).all()
        combs01 = torch.bitwise_and(
            torch.arange(2**self.event_size,
                         device=self._device).unsqueeze(-1),
            2**torch.arange(self.event_size, device=self._device))
        combs01 = (combs01 > 0).view((2**self.event_size, self.event_size))
        corners = sample[:, 0, :].view(
            (batch_shape.numel(), 1, self.event_size))\
            + sides.view((batch_shape.numel(), 1, self.event_size)) \
            * combs01
        result = (self.get_cdf(corners) *
                  (2*(combs01.count_nonzero(-1) % 2)-1)
                  ).sum(-1)
        if self.event_size % 2 == 0:
            result *= -1
        return result.view(batch_shape)

    def get_empirical_cdf(self,
                          count: int,
                          sample: torch.Tensor) -> torch.Tensor:
        """
        Generates 'count' many random samples, and uses them to evaluate the
        empirical distribution function on 'sample'. The input 'sample' is of
        shape batch_shape + event_shape and the output is of shape batch_shape.
        """

        batch_shape = sample.shape[:-1]
        assert sample.shape == batch_shape + self.event_shape
        sample = sample.view((batch_shape.numel(), self.event_size)
                             ).to(device=self._device)
        points = self.sample(torch.Size((count, ))
                             ).view(torch.Size((count, 1, self.event_size))
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
        batch_shape = rectangles.shape[:-2]
        assert rectangles.shape == batch_shape + (2,) + self.event_shape
        upper_bounds = rectangles.view((batch_shape.numel(), 2, 1,
                                        self.event_size)
                                       )[:, 1, :, :].to(device=self._device)
        lower_bounds = rectangles.view((batch_shape.numel(), 2, 1,
                                        self.event_size)
                                       )[:, 0, :, :].to(device=self._device)
        points = self.sample(torch.Size((count, ))
                             ).view(torch.Size((1, count, self.event_size))
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
        assert balls.shape == batch_shape + (self.event_size + 1, )
        balls = balls.view((batch_shape.numel(), self.event_size + 1))
        radius = balls[:, -1].view((batch_shape.numel(), 1))
        center = balls[:, :-1]
        center = center.view((batch_shape.numel(), 1,
                              self.event_size)
                             ).to(device=self._device)
        points = self.sample(torch.Size((count, ))
                             ).view(torch.Size((1, count, self.event_size))
                                    ).to(device=self._device)
        distance = (points - center).pow(2).sum(-1)
        result = (distance < radius.pow(2)).count_nonzero(-1)/count
        return result.view(batch_shape)

    def plot_empirical_pdf(self,
                           min_bound: float = -1.0,
                           max_bound: float = 1.0,
                           bins: int = 60,
                           count: int = 100000,
                           title: str = ""):
        """
        Takes count many samples from the distribution and plots the resulting
        histogram approximating the probability density of the distribution.
        This method assumes that the dimension of the distribution is one or two.
        """
        if title == "":
            plot_title = "Empirical PDF"
        else:
            plot_title = "Empirical PDF: " + title
        if self.event_size == 1:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().flatten().detach().numpy()
            pyplot.hist(sample,
                        bins=bins,
                        range=(min_bound, max_bound),
                        density=True)
            pyplot.title(plot_title)
            pyplot.show()
        elif self.event_size == 2:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().reshape((count, 2)).detach().numpy()
            pyplot.hist2d(sample[:, 0], sample[:, 1],
                          bins=bins,
                          range=((min_bound, max_bound),
                                 (min_bound, max_bound)),
                          density=True,
                          rasterized=True)
            pyplot.colorbar()
            pyplot.title(plot_title)
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_empirical_cdf(self,
                           min_bound: float = -1.0,
                           max_bound: float = 1.0,
                           bins: int = 120,
                           count: int = 100000,
                           title: str = ""):
        """
        Takes count many samples from the distribution and plots the resulting
        cumulative histogram approximating the cumulative distribution function.
        This method assumes that the dimension of the distribution is one.
        """
        if title == "":
            plot_title = "Empirical CDF"
        else:
            plot_title = "Empirical CDF: " + title
        if self.event_size == 1:
            sample = self.sample(torch.Size((count, )))
            sample = sample.cpu().flatten().detach().numpy()
            pyplot.hist(sample,
                        bins=bins,
                        range=(min_bound, max_bound),
                        density=True,
                        cumulative=True)
            pyplot.title(plot_title)
            pyplot.show()
        elif self.event_size == 2:
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
            pyplot.title(plot_title)
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_exact_pdf(self,
                       min_bound: float = -1.0,
                       max_bound: float = 1.0,
                       bins: int = 60,
                       title: str = ""):
        """
        Creates a grid of sample points and plots the corresponding probability
        density values as calculated by the log_prob method. This method assumes
        that the dimension of the distribution is one or two.
        """
        if title == "":
            plot_title = "Exact PDF"
        else:
            plot_title = "Exact PDF: " + title
        if self.event_size == 1:
            width = (max_bound - min_bound) / bins
            sample = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample = sample.view(torch.Size((bins,)) + self.event_shape)
            value = self.get_pdf(sample)
            pyplot.bar(
                x=sample.cpu().flatten().numpy(),
                height=value.cpu().flatten().detach().numpy(),
                width=width)
            pyplot.title(plot_title)
            pyplot.show()
        elif self.event_size == 2:
            width = (max_bound - min_bound) / bins
            sample1 = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
            sample2 = torch.stack(sample2, dim=-1).view(
                torch.Size((bins, bins)) + self.event_shape)
            value2 = self.get_pdf(sample2)
            pyplot.pcolormesh(
                sample1.cpu().numpy(),
                sample1.cpu().numpy(),
                value2.cpu().detach().numpy(),
                rasterized=True)
            pyplot.colorbar()
            pyplot.title(plot_title)
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_exact_cdf(self,
                       min_bound: float = -1.0,
                       max_bound: float = 1.0,
                       bins: int = 120,
                       title: str = ""):
        """
        Creates a grid of sample points and plots the corresponding cumulative
        distribution function values as calculated by the get_cdf method. This
        method assumes that the dimension of the distribution is one or two.
        """
        if title == "":
            plot_title = "Exact CDF"
        else:
            plot_title = "Exact CDF: " + title
        if self.event_size == 1:
            width = (max_bound - min_bound) / bins
            sample = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample = sample.view(torch.Size((bins,)) + self.event_shape)
            value = self.get_cdf(sample).detach()
            pyplot.bar(
                x=sample.cpu().flatten().numpy(),
                height=value.cpu().flatten().numpy(),
                width=width)
            pyplot.title(plot_title)
            pyplot.show()
        elif self.event_size == 2:
            width = (max_bound - min_bound) / bins
            sample1 = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
            sample2 = torch.stack(sample2, dim=-1).view(
                torch.Size((bins, bins)) + self.event_shape)
            value2 = self.get_cdf(sample2).detach()
            pyplot.pcolormesh(
                sample1.cpu().numpy(),
                sample1.cpu().numpy(),
                value2.cpu().numpy(),
                rasterized=True)
            pyplot.colorbar()
            pyplot.title(plot_title)
            pyplot.show()
        else:
            raise ValueError("invalid event size")

    def plot_pdf_from_cdf(self,
                          min_bound: float = -1.0,
                          max_bound: float = 1.0,
                          bins: int = 60,
                          title: str = ""):
        """
        Plots the approximate exact pdf using the exact cdf function.
        This method assumes that the dimension of the distribution
        is one or two.
        """
        if title == "":
            plot_title = "Approximate PDF"
        else:
            plot_title = "Approximate PDF: " + title
        if self._event_size == 1:
            width = (max_bound - min_bound) / bins
            sample = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample = sample.view((bins, self._event_size))
            rectangles = torch.stack(
                (sample - torch.tensor([0.5 * width]),
                 sample + torch.tensor([0.5 * width])),
                dim=-2)
            value = self.get_rectangle_prob(rectangles)
            value *= 1.0/float(width)
            pyplot.bar(
                x=sample.cpu().flatten().numpy(),
                height=value.cpu().flatten().detach().numpy(),
                width=width)
            pyplot.title(plot_title)
            pyplot.show()
        elif self._event_size == 2:
            width = (max_bound - min_bound) / bins
            sample1 = torch.linspace(
                min_bound + 0.5 * width,
                max_bound - 0.5 * width,
                bins,
                dtype=torch.float32,
                device=self._device)
            sample2 = torch.meshgrid([sample1, sample1], indexing="xy")
            sample2 = torch.stack(sample2, dim=-1).view(
                (bins, bins, self._event_size))
            rectangles2 = torch.stack(
                (sample2 - torch.tensor([0.5*width, 0.5*width]),
                 sample2 + torch.tensor([0.5*width, 0.5*width]),),
                dim=-2)
            value2 = self.get_rectangle_prob(rectangles2)
            value2 *= 1.0/float(width*width)
            pyplot.pcolormesh(
                sample1.cpu().numpy(),
                sample1.cpu().numpy(),
                value2.cpu().detach().numpy(),
                rasterized=True)
            pyplot.colorbar()
            pyplot.title(plot_title)
            pyplot.show()
        else:
            raise ValueError("invalid event size")
