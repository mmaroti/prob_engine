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


class MultiNormal(Distribution):
    def __init__(self,
                 means: torch.Tensor,
                 sdevs: torch.Tensor,
                 device: Optional[str] = None):
        assert means.shape == sdevs.shape

        Distribution.__init__(self, means.shape, device=device)
        self._means = means.to(dtype=torch.float32, device=self._device)
        self._sdevs = sdevs.to(dtype=torch.float32, device=self._device)

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


def test():
    dist = MultiNormal(
        torch.tensor(0.0),
        torch.tensor(1.0))
    print(dist.event_shape)
    # dist.plot_exact_density()
    dist.plot_sample_density()
    # dist.plot_sample_cumulative()
