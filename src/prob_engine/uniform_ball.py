# Copyright (C) 2023, Daniel Bezdany
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

class UniformBall(Distribution):
    def __init__(self,
                 center: torch.Tensor,
                 radius: torch.Tensor,
                 device: Optional[str] = None):
        
        assert radius.numel() == 1
        assert (radius >= 0).all()
        Distribution.__init__(self, center.shape, device=device)
        self._center = center.to(dtype=torch.float32, device=self._device)
        self._radius = radius.abs().flatten().to(
                                 dtype=torch.float32, device=self._device)

    @property
    def center(self) -> torch.Tensor:
        return self._center

    @property
    def radius(self) -> torch.Tensor:
        return self._radius

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._center
        yield self._radius

    def measure(self) -> torch.Tensor:
        if self.event_numel == 1:
            return self.radius * 2
        elif self.event_numel == 2:
            return self.radius.pow(2) * torch.pi
        elif self.event_numel == 3:
            return self.radius.pow(3) * 3 * torch.pi / 4
        else: 
            return ( self.radius.pow(self.event_numel) 
                    * torch.tensor(torch.pi).pow( self.event_numel/2.0 ) 
                    / torch.tensor( 1.0 + self.event_numel/2.0 ).lgamma().exp() )

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        directions = torch.normal( 0.0, 1.0,
                                  size = batch_shape + (self.event_numel,) )
        directions /= directions.pow(2).sum(-1).pow(0.5).unsqueeze(-1)
        rad = torch.rand( size = batch_shape 
                         ).pow( 1.0/self.event_numel ) * self.radius.abs()
        return self.center + (directions * rad.unsqueeze(-1)
                              ).view(batch_shape + self.event_shape)

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (0 < self.radius).all(), "No density function exists!"
        batch_shape = sample.shape[:-len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape
        inside = ( sample.view( batch_shape + (self.event_numel,) ) 
                  - self.center.flatten()
                  ).pow(2).sum(-1).pow(0.5) <= self.radius
        return inside/self.measure()
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log( self.get_pdf(sample) )

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        if (self.radius == 0).all():
            probs = (sample.view( batch_shape + (self.event_numel, ) )
                      >= self.center.flatten()
                      ).all(-1)
            return probs
        else:
            assert (0 < self.radius).all()
            batch_shape = sample.shape[:-len(self.event_shape)]
            assert sample.shape == batch_shape + self.event_shape

            if self.event_numel == 1:
                probs = torch.minimum(
                        torch.maximum(sample - self.center + self.radius,
                                    torch.tensor(0))
                        / self.measure(),
                        torch.tensor(1.0))
                return probs.view(batch_shape)
            
            elif self.event_numel == 2:
                """TODO: Improve efficiency, right now everything is
                calculated twice, then the unneeded one multiplied by 0,
                and the two added together at the end."""
                s = sample.view( ( batch_shape.numel(), self.event_numel ) )
                c = self.center.flatten()
                r = self.radius.flatten()
                """If the point is outside the circle in first quadrant,
                we use a different calculation for the area."""
                ext_q1 = torch.logical_and(
                    (s-c).pow(2).sum(-1) > r.pow(2),
                    (s > c).prod(-1) )
                
                corner_coord_diff = torch.maximum(
                    r.pow(2) - (c - s).pow(2), 
                    torch.tensor([0,0]) ).sqrt()
                corner_sides = torch.minimum(
                    corner_coord_diff 
                                + torch.maximum(
                                    torch.minimum(
                                        (s-c)[:,[1,0]],
                                        corner_coord_diff),
                                    -corner_coord_diff),
                    2*r  )
                corner_tri = corner_sides.prod(-1)/2
                corner_chord = torch.minimum(
                    corner_sides.pow(2).sum(-1).sqrt(),
                    2*r )
                corner_angle = 2 * ( corner_chord / (2 * r) ).arcsin()
                corner_segm = r.pow(2) * (
                    corner_angle - corner_angle.sin() ) / 2
                corner_area = corner_tri + corner_segm

                complement_angles = 2 * (corner_sides / (2 * r)).arcsin()
                complement_area = (
                    torch.pi * r.pow(2) - r.pow(2) * (
                        complement_angles 
                        - complement_angles.sin()
                        ).sum(-1) / 2
                    )
                area = ( ext_q1.logical_not() * corner_area
                        + ext_q1 * complement_area )
                return area.view(batch_shape)/self.measure()
            
            else:
                raise NotImplemented()
        


def test():
    dist1 = UniformBall(torch.tensor([0.25]), torch.tensor(0.25))
    print("Parameters", list(dist1.parameters))
    print("Volume of n-ball", dist1.measure())
    dist1.plot_exact_pdf()
    dist1.plot_empirical_pdf()
    dist1.plot_empirical_cdf()
    dist1.plot_exact_cdf()

    dist2 = UniformBall(torch.tensor([-0.5, -0.5]), torch.tensor(0.5))
    print("Parameters", list(dist2.parameters))
    print("Volume of n-ball", dist2.measure())
    dist2.plot_exact_pdf()
    dist2.plot_empirical_pdf()
    dist2.plot_empirical_cdf()
    dist2.plot_exact_cdf()
