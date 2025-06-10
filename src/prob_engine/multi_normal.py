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


class MultiNormal(Distribution):
    def __init__(self,
                 means: torch.Tensor,
                 sdevs: torch.Tensor,
                 device: Optional[str] = None):
        assert means.shape == sdevs.shape

        Distribution.__init__(self, means.shape, device=device)
        self._means = torch.nn.Parameter(
            means.to(dtype=torch.float32, device=self._device))
        self._sdevs = torch.nn.Parameter(
            sdevs.to(dtype=torch.float32, device=self._device))

    @property
    def means(self) -> torch.Tensor:
        return self._means

    @property
    def sdevs(self) -> torch.Tensor:
        return self._sdevs

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._means
        yield self._sdevs

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        standard = torch.normal(0.0, 1.0,
                                size=batch_shape + (self.event_numel, ),
                                device=self._device
                                ).view(batch_shape + self._event_shape)
        result = self._means + standard * self._sdevs.abs()
        return result

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self._sdevs.abs() > 0).all()
        # Could implement as sdev_i=0 meaning that i-th coordinate is fixed
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (self.event_numel, )
                             ).to(device=self._device)
        flat_means = self._means.flatten()
        flat_sdevs = self._sdevs.abs().flatten()

        coeff = torch.tensor(2 * torch.pi, device=self._device)
        coeff = coeff.pow(-self.event_numel/2.0)
        sqrdet = flat_sdevs.prod()
        exparg = sample - flat_means
        exparg = exparg.pow(2) / flat_sdevs.pow(2)
        exparg = -0.5 * exparg.sum(-1)
        return coeff * exparg.exp() / sqrdet

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self._sdevs.abs() > 0).all()
        # Could implement as sdev_i=0 meaning that i-th coordinate is fixed
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (self.event_numel, )
                             ).to(device=self._device)
        flat_means = self._means.flatten()
        flat_sdevs = self._sdevs.abs().flatten()

        coeff = torch.tensor(2 * torch.pi, device=self._device)
        sqrdetlog = flat_sdevs.prod().log()
        exparg = sample - flat_means
        exparg = exparg.pow(2) / flat_sdevs.pow(2)
        exparg = -0.5 * exparg.sum(-1)
        result = - self.event_numel * coeff.log() / 2.0 \
            - sqrdetlog + exparg
        return result

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self._sdevs.abs() > 0).all()
        # Could implement as sdev_i=0 meaning that i-th coordinate is fixed
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (self.event_numel, )
                             ).to(device=self._device)
        flat_means = self._means.flatten()
        flat_sdevs = self._sdevs.abs().flatten()
        """
        Currently, covariance matrix is assumed to be diagonal,
        which implies that the coordinates are independent.
        """
        sq2 = torch.tensor(2, device=self._device).sqrt()
        arg = sample - flat_means
        arg = arg / (flat_sdevs * sq2)
        result = 0.5 + 0.5 * torch.erf(arg)
        result = result.prod(-1)
        return result.view(batch_shape)


def test():
    dist1 = MultiNormal(torch.tensor([0.0]), torch.tensor([0.5]))
    print("Event shape", dist1.event_shape)
    print("Parameters", list(dist1.parameters))
    dist1.plot_empirical_pdf()
    dist1.plot_exact_pdf()
    dist1.plot_empirical_cdf()
    dist1.plot_exact_cdf()

    dist2 = MultiNormal(
        torch.tensor([0.0, 0.0]),
        0.1*torch.tensor([1.0, 0.3]))
    print("Event shape", dist2.event_shape)
    print("Parameters", list(dist2.parameters))
    dist2.plot_empirical_pdf()
    dist2.plot_exact_pdf()
    dist2.plot_empirical_cdf()
    dist2.plot_exact_cdf()
