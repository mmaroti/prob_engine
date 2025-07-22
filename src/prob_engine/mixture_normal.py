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
from torch.nn import Parameter

from .distribution import Distribution


class MixtureNormal(Distribution):
    def __init__(self,
                 means: torch.Tensor,
                 sdevs: torch.Tensor,
                 device: Optional[str] = None):
        assert means.dim() > 1
        assert means.shape == sdevs.shape

        Distribution.__init__(self, means.shape[1:], device=device)
        self._means = Parameter(means.to(dtype=torch.float32, device=self._device))
        self._sdevs = Parameter(sdevs.to(dtype=torch.float32, device=self._device))
        self._weights = Parameter(torch.rand(
                size=(means.shape[0],), dtype=torch.float32, device=self._device))
        
    @property
    def means(self) -> torch.Tensor:
        return self._means

    @property
    def sdevs(self) -> torch.Tensor:
        return self._sdevs
    
    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._means
        yield self._sdevs
        yield self._weights
    
    def reset_weights(self):
        weights = torch.rand(self._weights.shape, 
                       dtype=torch.float32, device=self._device)
        self._weights = Parameter(weights)
    
    def initialize_weights(self, weights: torch.Tensor):
        assert weights.numel() == self._weights.numel()
        assert weights.count_nonzero() > 0
        self._weights = Parameter(
            weights.view(self._weights.shape).to(
                dtype=torch.float32, device=self._device))
    
    def initialize_from_discrete(self, d: Distribution):
        from prob_engine import discrete
        if d.isinstance(discrete.Discrete):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def add_normals(self, means: torch.Tensor, sdevs: torch.Tensor):
        assert means.shape == sdevs.shape
        if means.shape == self._event_shape:
            means = torch.cat((self._means, means), 0).to(
                dtype=torch.float32, device=self._device)
            sdevs = torch.cat((self._sdevs, sdevs), 0).to(
                dtype=torch.float32, device=self._device)
            w_sum = self._weights.abs().sum()
            new_weight = torch.tensor(w_sum/(w_sum+1))
            weights = torch.cat((self._weights, new_weight)).to(
                dtype=torch.float32, device=self._device)
            self._means = Parameter(means)
            self._sdevs = Parameter(sdevs)
            self._weights = Parameter(weights)
        else:
            batch_shape = means.shape[:-len(self._event_shape)]
            assert means.shape == batch_shape + self._event_shape
            means = means.view((batch_shape.numel(),)+self._event_shape
                            ).to(dtype=torch.float32,device=self._device)
            sdevs = sdevs.view((batch_shape.numel(),)+self._event_shape
                               ).to(dtype=torch.float32,device=self._device)
            w_sum = self._weights.abs().sum()
            fill_val = w_sum.item()/(w_sum.item()+batch_shape.numel())
            weights = torch.full(size = (batch_shape.numel(),),
                                 fill_value=fill_val,
                                 dtype=torch.float32, device=self._device)
            self._means = Parameter(torch.cat((self._means, means), 0))
            self._sdevs = Parameter(torch.cat((self._means, sdevs), 0))
            self._weights = Parameter(torch.cat((self._weights, weights),0))
    
    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        selection = torch.multinomial(
            self._weights.abs(),
            batch_shape.numel(),
            replacement=True).view(batch_shape)
        standard = torch.normal(0.0, 1.0,
                                size=batch_shape + (self.event_numel, ),
                                device=self._device
                                ).view(batch_shape + self._event_shape)
        result = self._means[selection] + standard * self._sdevs.abs()[selection]
        return result
    
    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()