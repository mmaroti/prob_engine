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


class Restriction(Distribution):
    def __init__(self,
                 distribution: Distribution,
                 domain: torch.Tensor,
                 device: Optional[str] = None):
        assert domain.shape == (2,) + distribution.event_shape
        assert (domain[0,:] <= domain[1,:]).all().item()
        domain_weight = distribution.get_rectangle_prob(domain).item()
        assert domain_weight > 0
        Distribution.__init__(
            self, distribution._event_size, device=device)
        self._raw_distribution = distribution
        self._domain = domain
#        self._domain_weight = domain_weight

    @property
    def raw_distribution(self) -> Distribution:
        return self._raw_distribution
    
    @property
    def domain(self) -> torch.Tensor:
        return self._domain
    
    @property
    def domain_weight(self) -> torch.Tensor:
        return self._raw_distribution.get_rectangle_prob(self._domain)
    
    @property
    def domain_measure(self) -> torch.Tensor:
        return (self._domain[1,:]-self._domain[0,:]).relu().prod()
    
    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self._raw_distribution.parameters
    
    def reset_restriction(self, new_domain: torch.Tensor):
        assert new_domain.shape == (2,) + self.raw_distribution.event_shape
        assert (new_domain[0,:] <= new_domain[1,:]).all()
        new_domain_weight = self._raw_distribution.get_rectangle_prob(new_domain).item()
        assert new_domain_weight > 0
        self._domain = new_domain
#        self._domain_weight = new_domain_weight

    def sample(self, batch_shape: torch.Size = torch.Size(), max_iterations: int = 0) -> torch.Tensor:
        if max_iterations == 0:
            max_iterations = 10 * batch_shape.numel()
        assert batch_shape.numel() > 0
        assert max_iterations > 0
        desired_count = batch_shape.numel()
        desired_shape = torch.Size([desired_count])
        temp = self._raw_distribution.sample(desired_shape)
        total = temp.shape[0]
        temp = temp[torch.logical_and(
            (temp >= self._domain[0,:].unsqueeze(0)).prod(-1),
            (temp <= self._domain[1,:].unsqueeze(0)).prod(-1))]
        filtered = temp.shape[0]
        if filtered < desired_count and total < max_iterations:
            while filtered < desired_count and total < max_iterations:
                this_count = desired_count - filtered
                temp1 = self._raw_distribution.sample(torch.Size([this_count]))
                total += this_count
                temp1 = temp1[torch.logical_and(
                    (temp1 >= self._domain[0,:].unsqueeze(0)).prod(-1),
                    (temp1 <= self._domain[1,:].unsqueeze(0)).prod(-1))]
                filtered += temp1.shape[0]
                temp = torch.cat((temp,temp1), 0)
        result = temp.view((filtered,) + self._raw_distribution.event_shape)
        return result

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-1]
        assert sample.shape == batch_shape + self.event_shape
        flat_sample = sample.view((batch_shape.numel(),self.event_size)).to(
                                    dtype=torch.float32, device=self._device)
        flat_domain = self.domain.view((2, self.event_size)).to(
            dtype=torch.float32, device=self._device)
        inside = torch.logical_and(flat_sample >= flat_domain[0,:].unsqueeze(0),
                                    flat_sample <= flat_domain[1,:].unsqueeze(0)).prod(-1)
        raw_pdf = self.raw_distribution.get_pdf(sample) 
        result = raw_pdf * inside.view(raw_pdf.shape)
        result *= (1/self.domain_weight)
        return result

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-1]
        assert sample.shape == batch_shape + self.event_shape
        flat_sample = sample.view((batch_shape.numel(), self.event_size)).to(
            dtype = torch.float32, device=self._device)
        flat_domain = self.domain.view((2, self.event_size)).to(
            dtype=torch.float32, device=self._device)
        flat_upper = torch.ones((flat_sample.shape[0],1))*flat_domain[-1,:].unsqueeze(0)
        flat_lower = torch.ones((flat_sample.shape[0],1))*flat_domain[0,:].unsqueeze(0)
        temp = torch.cat((flat_sample.unsqueeze(-2),
                                       flat_lower.unsqueeze(-2)), -2).max(dim=-2)[0]
        temp = torch.cat((temp.unsqueeze(-2), flat_upper.unsqueeze(-2)),-2).min(dim=-2)[0]
        rectangles = torch.cat(
            (flat_lower.view((batch_shape.numel(),1)+self.event_shape),
             temp.view((batch_shape.numel(),1)+self.event_shape)), 1)
        result = self.raw_distribution.get_rectangle_prob(
            rectangles.view(batch_shape + (2,) + self.event_shape))
        result *= (1/self.domain_weight)
        return result


def test():
    from .misc.uniform_ball import UniformBall
    raw_dist = UniformBall(torch.tensor([0.0,0.0]), torch.tensor([1]))
    restricted_dist = Restriction(raw_dist, torch.tensor([[0,0],[1,1]]))

    print("Distributions:", raw_dist, restricted_dist)
    print("Event shapes:", raw_dist.event_shape, restricted_dist.event_shape)
    print("Parameters:", list(raw_dist.parameters), list(restricted_dist.parameters))
    raw_dist.plot_exact_pdf()
    restricted_dist.plot_exact_pdf()
    raw_dist.plot_empirical_pdf()
    restricted_dist.plot_empirical_pdf()
    raw_dist.plot_empirical_cdf()
    restricted_dist.plot_empirical_cdf()
    raw_dist.plot_exact_cdf()
    restricted_dist.plot_exact_cdf()
    inf_point = torch.full(raw_dist.event_shape,torch.inf)
    ninf_point = -inf_point
    pinf_point = torch.tensor([torch.inf, 0.0])
    ppinf_point = torch.tensor([torch.inf, 0.25])
    inf_points = torch.stack((inf_point, ninf_point, pinf_point, ppinf_point), 0)
    print("Evaluations of get_cdfs at Infinity:",
          raw_dist.get_cdf(inf_points),
          restricted_dist.get_cdf(inf_points))