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
        self._means = Parameter(
            means.to(dtype=torch.float32, device=self._device))
        self._sdevs = Parameter(
            sdevs.to(dtype=torch.float32, device=self._device))
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
    
    @property
    def count(self) -> int:
        return self._weights.numel()

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

    def initialize_from_distribution(self, d: Distribution):
        from prob_engine import discrete
        if isinstance(d, discrete.Discrete):
            dmatrix = torch.cdist(d._atoms, d._atoms, p=2)
            if dmatrix.max() <= 0:
                min_distance = torch.tensor(1.0)
            else:
                dmatrix = dmatrix + (dmatrix <= 0) * dmatrix.sum()
                min_distance = dmatrix.min()
            means = d.atoms
            sdevs = torch.empty(means.shape)
            sdevs = torch.fill(sdevs, min_distance/3.0)
            self._means = means
            self._sdevs = sdevs
            self._weights = d._weights
        else:
            raise NotImplementedError()

    def add_normals(self, means: torch.Tensor, sdevs: torch.Tensor):
        assert means.shape == sdevs.shape
        if means.shape == self._event_shape:
            means = means.view((1,) + self._event_shape).to(
                dtype=torch.float32, device=self._device)
            sdevs = sdevs.view((1,) + self._event_shape).to(
                dtype=torch.float32, device=self._device)
        batch_shape = means.shape[:-len(self._event_shape)]
        assert means.shape == batch_shape + self._event_shape
        means = means.view((batch_shape.numel(),)+self._event_shape
                            ).to(dtype=torch.float32, device=self._device)
        sdevs = sdevs.view((batch_shape.numel(),)+self._event_shape
                            ).to(dtype=torch.float32, device=self._device)
        w_sum = self._weights.abs().sum()
        fill_val = float(w_sum.item())/float(self.count)
        weights = torch.full(size=(batch_shape.numel(),),
                                fill_value=fill_val,
                                dtype=torch.float32, device=self._device)
        self._means = Parameter(torch.cat((self._means, means), 0))
        self._sdevs = Parameter(torch.cat((self._means, sdevs), 0))
        self._weights = Parameter(torch.cat((self._weights, weights), 0))

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        selection = torch.multinomial(
            self._weights.abs(),
            batch_shape.numel(),
            replacement=True).view(batch_shape)
        standard = torch.normal(0.0, 1.0,
                                size=batch_shape + (self.event_numel, ),
                                device=self._device
                                ).view(batch_shape + self._event_shape)
        result = self._means[selection] + \
            standard * self._sdevs.abs()[selection]
        return result

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self._sdevs.abs() > 0).all()
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (1, self.event_numel)
                             ).to(device=self._device)
        flat_means = self._means.view((self.count,self.event_numel))
        flat_sdevs = self._sdevs.abs().view((self.count,self.event_numel))

        coeff = torch.tensor(2 * torch.pi, device=self._device)
        coeff = coeff.pow(-self.event_numel/2.0)
        sqrdet = flat_sdevs.prod(-1)
        exparg = sample - flat_means
        exparg = exparg.pow(2) / flat_sdevs.pow(2)
        exparg = -0.5 * exparg.sum(-1)
        temp = coeff * exparg.exp() / sqrdet
        temp = temp * self._weights.abs()
        temp = temp.sum(-1)
        temp *= 1.0/self._weights.abs().sum(-1)
        return temp

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return self.get_pdf(sample).log()

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self._sdevs.abs() > 0).all()
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view(batch_shape + (1, self.event_numel)
                             ).to(device=self._device)
        flat_means = self._means.view((self.count,self.event_numel))
        flat_sdevs = self._sdevs.abs().view((self.count,self.event_numel))

        sq2 = torch.tensor(2, device=self._device).sqrt()
        arg = sample - flat_means
        arg = arg / (flat_sdevs * sq2)
        result = 0.5 + 0.5 * torch.erf(arg)
        result = result.prod(-1)
        result = (result * self._weights.abs()).sum(-1)
        result *= 1.0/self._weights.sum()
        return result.view(batch_shape)
    

def test():
    from .normal import Normal
    from .mixture import Mixture
    from .discrete import Discrete

    n1 = Normal(torch.tensor([-0.5]), torch.tensor([0.25]))
    n2 = Normal(torch.tensor([0.5]), torch.tensor([0.25]))
    mix = Mixture(list((n1, n2)))
    mixn = MixtureNormal(torch.tensor([[-0.5], [0.5]]), 
                       torch.tensor([[0.25],[0.25]]))
    mixn.initialize_weights(mix._weights)

    mix.plot_exact_pdf()
    mixn.plot_exact_pdf()
    mix.plot_exact_cdf()
    mixn.plot_exact_cdf()
    print(mixn.sample(torch.Size((2, 3))))

    nn1 = Normal(torch.tensor([-0.5, -0.5]),
                 torch.tensor([0.25, 0.15]))
    nn2 = Normal(torch.tensor([0.5, 0.5]),
                 torch.tensor([0.15, 0.25]))
    mix2 = Mixture(list((nn1, nn2)))
    mixn2 = MixtureNormal(
        torch.tensor([[-0.5,-0.5],[0.5,0.5]]),
        torch.tensor([[0.25,0.15],[0.15,0.25]]))
    mixn2.initialize_weights(mix2._weights)

    mix2.plot_exact_pdf()
    mixn2.plot_exact_pdf()
    mix2.plot_exact_cdf()
    mixn2.plot_exact_cdf()
    print(mixn2.sample(torch.Size((2, 3))))

    disc = Discrete(torch.tensor([[-0.6,-0.2],[-0.3,-0.7],[0.1,0.1],[0.2,0.8]]))
    discN = MixtureNormal(torch.tensor([[0,0]]), torch.tensor([[1.0,1.0]]))
    discN.plot_exact_pdf()
    discN.plot_exact_cdf()
    discN.initialize_from_distribution(disc)
    discN.plot_exact_pdf()
    discN.plot_exact_cdf()
    disc.plot_exact_cdf()