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

class UniformShell(Distribution):
    def __init__(self,
                 center: torch.Tensor,
                 radius1: torch.Tensor,
                 radius2: torch.Tensor,
                 device: Optional[str] = None):
        
        assert radius1.numel() == 1 and radius2.numel() == 1
        assert torch.logical_and( radius1 >= 0, radius2 >= 0 ).all()
        Distribution.__init__(self, center.shape, device=device)
        self._center = center.to(dtype=torch.float32, device=self._device)
        self._radius1 = radius1.flatten().to(dtype=torch.float32, device=self._device)
        self._radius2 = radius2.flatten().to(dtype=torch.float32, device=self._device)

    @property
    def center(self) -> torch.Tensor:
        return self._center

    @property
    def radius1(self) -> torch.Tensor:
        return self._radius1
    
    @property
    def radius2(self) -> torch.Tensor:
        return self._radius2

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._center
        yield self._radius1
        yield self._radius2

    def measure(self) -> torch.Tensor:
        if self.event_numel == 1:
            diff = self._radius2.abs() - self._radius1.abs()
            return diff.abs() * 2
        elif self.event_numel == 2:
            diff = self._radius2.abs().pow(2) \
                    - self._radius1.abs().pow(2)
            return diff.abs() * torch.pi
        elif self.event_numel == 3:
            diff = self._radius2.abs().pow(3) \
                    - self._radius1.abs().pow(3)
            return diff.abs() * 3 * torch.pi / 4
        else: 
            diff = self._radius1.abs().pow(self.event_numel) \
                    - self._radius2.abs().pow(self.event_numel)
            c1 = torch.tensor( torch.pi, device=self._device
                              ).pow(self.event_numel/2.0)
            c2 = torch.tensor( 1.0 + self.event_numel/2.0,
                              device=self._device ).lgamma().exp()
            return diff.abs() * c1 / c2

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        directions = torch.normal( 0.0, 1.0,
                                  size = batch_shape + (self.event_numel, ),
                                  device = self._device )
        directions /= directions.pow(2).sum(-1).sqrt().unsqueeze(-1)
        diff = self._radius1.abs().pow(self.event_numel) \
                   - self._radius2.abs().pow(self.event_numel)
        rad = torch.rand( size = batch_shape, device=self._device ) \
                * diff.abs() \
                + torch.minimum(
                        self._radius1.abs(),
                        self._radius2.abs()
                        ).pow(self.event_numel)
        rad = rad.pow(1.0/self.event_numel)
        return self._center + (directions * rad.unsqueeze(-1)
                              ).view(batch_shape + self._event_shape)

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self.measure() >  0).all()
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view( batch_shape + (self.event_numel, )
                             ).to(device=self._device)
        distance = (sample - self._center.flatten()).pow(2).sum(-1).sqrt()
        inside = torch.logical_and(
                    distance >= torch.minimum(
                            self._radius1.flatten().abs(),
                            self._radius2.flatten().abs()),
                    distance <= torch.maximum(
                            self._radius1.flatten().abs(), 
                            self._radius2.flatten().abs()))    
        return inside/self.measure()
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log( self.get_pdf(sample) )

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view( batch_shape + (self.event_numel, )
                             ).to(device=self._device)
        if torch.logical_and(
            self._radius1 == 0,
            self._radius2 == 0).all():
            above = self._center.flatten() <= sample
            return above.all(-1).view(batch_shape)
        else:
            if self.event_numel == 1:
                if (self._radius1.abs() == self._radius2.abs()).all():
                    common_r = self._radius1.flatten().abs()
                    probs = sample >= self._center - common_r \
                            + sample >= self._center + common_r
                    return probs.view(batch_shape)/2.0
                elif (self._radius1.flatten() == 0).logical_or(
                        self._radius2.flatten() == 0).all():
                    nonz_r = self._radius1.flatten().abs() \
                                + self._radius2.flatten().abs() 
                    probs = torch.minimum(
                            (sample - self._center + nonz_r).relu(),
                            self.measure())
                    return probs.view(batch_shape)/self.measure()
                else:
                    large_r = torch.maximum(
                                        self._radius1.flatten().abs(),
                                        self._radius2.flatten().abs() )
                    small_r = torch.minimum(
                                        self._radius1.flatten().abs(),
                                        self._radius2.flatten().abs() )
                    larger = torch.minimum(
                                ( sample 
                                    - self._center.flatten() 
                                    + large_r ).relu(),
                                2 * large_r)
                    smaller = torch.minimum(
                                ( sample
                                    - self._center.flatten() 
                                    + small_r ).relu(),
                                2 * small_r)
                    diff = larger - smaller
                    return diff.view(batch_shape) / self.measure()
            elif self.event_numel == 2:
                if torch.logical_or(
                        self._radius1 == 0,
                        self._radius2 == 0 ).all():
                    sam = sample.view( ( batch_shape.numel(), self.event_numel )
                                ).to(device=self._device)
                    cen = self._center.flatten()
                    rad = self._radius1.flatten().abs() + self._radius2.flatten().abs()
                    """If the point is outside the circle in first quadrant,
                    we use a different calculation for the area."""
                    c = (sam >= cen + rad).all(-1)
                    b = torch.logical_and(
                            (sam > cen).all(-1),
                            (sam-cen).pow(2).sum(-1) > rad.pow(2)
                            ).logical_and(c.logical_not())
                    a = b.logical_or(c).logical_not()

                    diffs = ( rad.pow(2)-(cen - sam).pow(2) ).relu().sqrt()
                    sides = torch.minimum(
                        diffs + torch.maximum(
                            torch.minimum( (sam-cen)[:,[1,0]], diffs),
                            -diffs),  2*rad)
                    chord_a = torch.minimum( sides[a].pow(2).sum(-1).sqrt(), 2*rad )
                    angle_a = 2 * ( chord_a / (2* rad) ).arcsin()
                    area_a = sides[a].prod(-1)/2 + \
                                rad.pow(2) * ( angle_a - angle_a.sin() ) / 2
                    angles_b = 2 * (sides[b] / (2 * rad)).arcsin()
                    area_b = torch.pi - ( angles_b - angles_b.sin()).sum(-1) / 2
                    area_b = rad.pow(2) * area_b
                    area = torch.zeros(batch_shape.numel())
                    area[c] = self.measure()
                    area[a] = area_a
                    area[b] = area_b
                    return area.view(batch_shape)/self.measure()
                elif (self._radius1.abs() != self._radius2.abs()).all():
                    sam = sample.view( ( batch_shape.numel(), self.event_numel )
                                ).to(device=self._device)
                    cen = self._center.flatten()
                    rad1 = self._radius1.flatten().abs()
                    rad2 = self._radius2.flatten().abs()
                    """If the point is outside the circle in first quadrant,
                    we use a different calculation for the area."""
                    c1 = (sam >= cen + rad1).all(-1)
                    b1 = torch.logical_and(
                            (sam > cen).all(-1),
                            (sam - cen).pow(2).sum(-1) > rad1.pow(2)
                            ).logical_and(c1.logical_not())
                    a1 = b1.logical_or(c1).logical_not()

                    diffs1 = ( rad1.pow(2) - (cen - sam).pow(2) ).relu().sqrt()
                    sides1 = torch.minimum(
                        diffs1 + torch.maximum(
                            torch.minimum( (sam-cen)[:,[1,0]], diffs1),
                            -diffs1),  2*rad1)
                    chord_a1 = torch.minimum( sides1[a1].pow(2).sum(-1).sqrt(), 2*rad1 )
                    angle_a1 = 2 * ( chord_a1 / (2* rad1) ).arcsin()
                    area_a1 = sides1[a1].prod(-1)/2 + \
                                rad1.pow(2) * ( angle_a1 - angle_a1.sin() ) / 2
                    angles_b1 = 2 * (sides1[b1] / (2 * rad1)).arcsin()
                    area_b1 = torch.pi - ( angles_b1 - angles_b1.sin()).sum(-1) / 2
                    area_b1 = rad1.pow(2) * area_b1
                    area1 = torch.zeros(batch_shape.numel())
                    area1[c1] = rad1.pow(2)*torch.pi
                    area1[a1] = area_a1
                    area1[b1] = area_b1

                    c2 = (sam >= cen + rad2).all(-1)
                    b2 = torch.logical_and(
                            (sam > cen).all(-1).logical_and(c2.logical_not()),
                            (sam - cen).pow(2).sum(-1) > rad2.pow(2) )
                    a2 = b2.logical_or(c2).logical_not()

                    diffs2 = ( rad2.pow(2) - (cen - sam).pow(2) ).relu().sqrt()
                    sides2 = torch.minimum(
                        diffs2 + torch.maximum(
                            torch.minimum( (sam-cen)[:,[1,0]], diffs2),
                            -diffs2),  2*rad2)
                    chord_a2 = torch.minimum( sides2[a2].pow(2).sum(-1).sqrt(), 2*rad2 )
                    angle_a2 = 2 * ( chord_a2 / (2* rad2) ).arcsin()
                    area_a2 = sides2[a2].prod(-1)/2 + \
                                rad2.pow(2) * ( angle_a2 - angle_a2.sin() ) / 2
                    angles_b2 = 2 * (sides2[b2] / (2 * rad2)).arcsin()
                    area_b2 = torch.pi - ( angles_b2 - angles_b2.sin()).sum(-1) / 2
                    area_b2 = rad2.pow(2) * area_b2
                    area2 = torch.zeros(batch_shape.numel())
                    area2[c2] = rad2.pow(2)*torch.pi
                    area2[a2] = area_a2
                    area2[b2] = area_b2
                    area = (area1 - area2).abs()
                    return area.view(batch_shape) / self.measure()
                else:
                    """The case when the shell is a circle."""
                    assert (self._radius1.abs() == self._radius2.abs()).all()
                    sam = sample.view( ( batch_shape.numel(), self.event_numel )
                                ).to(device=self._device)
                    cen = self._center.flatten()
                    rad = self._radius1.flatten().abs()
                    """If the point is outside the circle in first quadrant,
                    we use a different calculation for the area."""
                    c = (sam >= cen + rad).all(-1)
                    b = torch.logical_and(
                            (sam > cen).all(-1),
                            (sam - cen).pow(2).sum(-1) > rad.pow(2)
                            ).logical_and(c.logical_not())
                    a = b.logical_or(c).logical_not()
                    diffs = ( rad.pow(2)-(cen - sam).pow(2) ).relu().sqrt()
                    sides = torch.minimum(
                        diffs + torch.maximum(
                            torch.minimum( (sam-cen)[:,[1,0]], diffs),
                            -diffs),  2*rad)
                    chord_a = torch.minimum( sides[a].pow(2).sum(-1).sqrt(), 2*rad )
                    angle_a = 2 * ( chord_a / (2* rad) ).arcsin()
                    arc_a = angle_a
                    angles_b = 2 * (sides[b] / (2 * rad)).arcsin()
                    arc_b = 2 * torch.pi - angles_b.sum(-1)
                    arc = torch.zeros(batch_shape.numel())
                    arc[c] = 2*torch.pi
                    arc[a] = arc_a
                    arc[b] = arc_b
                    return arc.view(batch_shape)/(2*torch.pi)
                


def test():
    dist1 = UniformShell(torch.tensor([0.25]), torch.tensor(0.25), torch.tensor(0.5))
    print("Parameters", list(dist1.parameters))
    print("Volume of hyperspherical shell", dist1.measure())
    dist1.plot_exact_pdf()
    dist1.plot_empirical_pdf()
    dist1.plot_empirical_cdf()
    dist1.plot_exact_cdf()

    dist2 = UniformShell(torch.tensor([-0.25, -0.25]), torch.tensor(0.25), torch.tensor(0.5))
    print("Parameters", list(dist2.parameters))
    print("Volume of hyperspherical shell", dist2.measure())
    dist2.plot_exact_pdf()
    dist2.plot_empirical_pdf()
    dist2.plot_empirical_cdf()
    dist2.plot_exact_cdf()

    dist3 = UniformShell(torch.tensor([-0.25, -0.25]), torch.tensor(0.45), torch.tensor(0.45))
    print("Parameters", list(dist3.parameters))
    print("Volume of hyperspherical shell", dist3.measure())
    #dist3.plot_exact_pdf()
    dist3.plot_empirical_pdf()
    dist3.plot_empirical_cdf()
    dist3.plot_exact_cdf()

    

