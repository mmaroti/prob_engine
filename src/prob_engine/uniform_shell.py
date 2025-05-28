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
        
        assert ( radius1.numel() == 1 and radius2.numel() == 1 )
        assert ( (radius1 >= 0) * (radius2 >= 0) ).all()
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
            return (self.radius2.flatten().abs()
                     - self.radius1.flatten().abs() ).abs() * 2
        elif self.event_numel == 2:
            return (self.radius1.flatten().pow(2)
                    -self.radius2.flatten().pow(2)).abs() * torch.pi
        elif self.event_numel == 3:
            return ( (self.radius1.flatten().abs().pow(3) 
                      - self.radius2.flatten().abs().pow(3)).abs()
                    * 3 * torch.pi / 4 )        
        else: 
            return ( ( self.radius1.flatten().abs().pow(self.event_numel) 
                       - self.radius2.flatten().abs().pow(self.event_numel)  ).abs()
                    * torch.tensor(torch.pi).pow( self.event_numel/2.0 ) 
                    / torch.tensor( 1.0 + self.event_numel/2.0 ).lgamma().exp()
                    )

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        directions = torch.normal( 0.0, 1.0,
                                  size = batch_shape + (self.event_numel, ) )
        directions /= directions.pow(2).sum(-1).pow(0.5).unsqueeze(-1)
        rad = ( torch.rand( size = batch_shape )
                * (self.radius1.abs().pow(self.event_numel) 
                   - self.radius2.abs().pow(self.event_numel)).abs()
                + torch.minimum(
                        self.radius1.abs(),
                        self.radius2.abs()
                        ).pow(self.event_numel)
                ).pow(1.0/self.event_numel)
        return self.center + (directions * rad.unsqueeze(-1)
                              ).view(batch_shape + self.event_shape)

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (self.measure() >  0).all()
        
        batch_shape = sample.shape[:-len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape
        distance = ( sample.view( batch_shape + (self.event_numel, ) )
                - self.center.flatten()
                ).pow(2).sum(-1).pow(0.5)
        inside = torch.logical_and(
                (distance >= torch.minimum(
                        self.radius1.flatten().abs(),
                        self.radius2.flatten().abs())),
                (distance <= torch.maximum(
                        self.radius1.flatten().abs(), 
                        self.radius2.flatten().abs())))    
        return inside/self.measure()
    
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log( self.get_pdf(sample) )

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self.event_shape)]
        assert sample.shape == batch_shape + self.event_shape
        if torch.logical_and(
            self.radius1 == 0,
            self.radius2 == 0).all():
                return (self.center <= sample).view(
                    batch_shape + (self.event_numel, ) ).prod(-1)
        else:
            if self.event_numel == 1:
                if (self.radius1.abs() == self.radius2.abs()).all():
                    probs = (
                        sample >= self.center - self.radius1.flatten().abs()
                        + sample >= self.center + self.radius1.flatten().abs()
                    )
                    return probs.view(batch_shape)/2.0
                elif (self.radius1 * self.radius2 == 0).all():
                    probs = torch.minimum(
                            torch.maximum(sample - self.center 
                                          + self.radius1.flatten().abs()
                                          + self.radius2.flatten().abs(),
                                        torch.tensor(0))
                            / self.measure(),
                            torch.tensor(1.0))
                    return probs.view(batch_shape)
                else:
                    large_r = torch.maximum(
                                        self.radius1.flatten().abs(),
                                        self.radius2.flatten().abs() )
                    small_r = torch.minimum(
                                        self.radius1.flatten().abs(),
                                        self.radius2.flatten().abs() )
                    larger = torch.minimum(
                            torch.maximum(sample.view( 
                                (batch_shape.numel(),self.event_numel))
                                - self.center.flatten() 
                                + large_r, torch.tensor(0) ),
                            2*large_r)
                    smaller = torch.minimum(
                            torch.maximum(sample.view( 
                                (batch_shape.numel(),self.event_numel))
                                - self.center.flatten() 
                                + small_r, torch.tensor(0) ),
                            2*small_r)
                    return ( ( larger - smaller) / self.measure()
                            ).view(batch_shape)
            elif self.event_numel == 2:
                if torch.logical_or(
                        self.radius1 == 0,
                        self.radius2 == 0 ).all():
                    """TODO: Improve efficiency, right now everything is
                    calculated twice, then the unneeded one multiplied by 0,
                    and the two added together at the end."""
                    s = sample.view( ( batch_shape.numel(), self.event_numel ) )
                    c = self.center.flatten()
                    rR = self.radius1.flatten().abs() + self.radius2.flatten().abs()
                    """If the point is outside the circle in first quadrant,
                    we use a different calculation for the area."""
                    ext_q1_rR = torch.logical_and(
                        (s-c).pow(2).sum(-1) > rR.pow(2),
                        (s > c).prod(-1) )
                    
                    corner_coord_diff_rR = torch.maximum(
                        rR.pow(2) - (c - s).pow(2), 
                        torch.tensor([0,0]) ).sqrt()
                    corner_sides_rR = torch.minimum(
                        corner_coord_diff_rR 
                                    + torch.maximum(
                                        torch.minimum(
                                            (s-c)[:,[1,0]],
                                            corner_coord_diff_rR),
                                        -corner_coord_diff_rR),
                        2*rR  )
                    corner_tri_rR = corner_sides_rR.prod(-1)/2
                    corner_chord_rR = torch.minimum(
                        corner_sides_rR.pow(2).sum(-1).sqrt(),
                        2*rR )
                    corner_angle_rR = 2 * ( corner_chord_rR / (2 * rR) ).arcsin()
                    corner_segm_rR = rR.pow(2) * (
                        corner_angle_rR - corner_angle_rR.sin() ) / 2
                    corner_area_rR = corner_tri_rR + corner_segm_rR

                    complement_angles_rR = 2 * (corner_sides_rR / (2 * rR)).arcsin()
                    complement_area_rR = (
                        torch.pi * rR.pow(2) - rR.pow(2) * (
                            complement_angles_rR 
                            - complement_angles_rR.sin()
                            ).sum(-1) / 2
                        )
                    area_rR = ( ext_q1_rR.logical_not() * corner_area_rR
                            + ext_q1_rR * complement_area_rR )
                    return area_rR.view(batch_shape)/self.measure()
                elif (self.radius1.abs() != self.radius2.abs()).all():
                    """TODO: Improve efficiency, right now everything is
                    calculated twice, then the unneeded one multiplied by 0,
                    and the two added together at the end."""
                    s = sample.view( ( batch_shape.numel(), self.event_numel ) )
                    c = self.center.flatten()
                    r1 = torch.minimum(
                        self.radius1.flatten().abs(),
                        self.radius2.flatten().abs() )
                    r2 = torch.maximum(
                        self.radius1.flatten().abs(),
                        self.radius2.flatten().abs() )
                    
                    ext_q1_r1 = torch.logical_and(
                        (s-c).pow(2).sum(-1) > r1.pow(2),
                        (s > c).prod(-1) )
                    
                    corner_coord_diff_r1 = torch.maximum(
                        r1.pow(2) - (c - s).pow(2), 
                        torch.tensor([0,0]) ).sqrt()
                    corner_sides_r1 = torch.minimum(
                        corner_coord_diff_r1 
                                    + torch.maximum(
                                        torch.minimum(
                                            (s-c)[:,[1,0]],
                                            corner_coord_diff_r1),
                                        -corner_coord_diff_r1),
                        2*r1  )
                    corner_tri_r1 = corner_sides_r1.prod(-1)/2
                    corner_chord_r1 = torch.minimum(
                        corner_sides_r1.pow(2).sum(-1).sqrt(),
                        2*r1 )
                    corner_angle_r1 = 2 * ( corner_chord_r1 / (2 * r1) ).arcsin()
                    corner_segm_r1 = r1.pow(2) * (
                        corner_angle_r1 - corner_angle_r1.sin() ) / 2
                    corner_area_r1 = corner_tri_r1 + corner_segm_r1

                    complement_angles_r1 = 2 * (corner_sides_r1 / (2 * r1)).arcsin()
                    complement_area_r1 = (
                        torch.pi * r1.pow(2) - r1.pow(2) * (
                            complement_angles_r1 
                            - complement_angles_r1.sin()
                            ).sum(-1) / 2
                        )
                    area_r1 = ( ext_q1_r1.logical_not() * corner_area_r1
                            + ext_q1_r1 * complement_area_r1 )

                    ext_q1_r2 = torch.logical_and(
                        (s-c).pow(2).sum(-1) > r2.pow(2),
                        (s > c).prod(-1) )
                    
                    corner_coord_diff_r2 = torch.maximum(
                        r2.pow(2) - (c - s).pow(2), 
                        torch.tensor([0,0]) ).sqrt()
                    corner_sides_r2 = torch.minimum(
                        corner_coord_diff_r2 
                                    + torch.maximum(
                                        torch.minimum(
                                            (s-c)[:,[1,0]],
                                            corner_coord_diff_r2),
                                        -corner_coord_diff_r2),
                        2*r2  )
                    corner_tri_r2 = corner_sides_r2.prod(-1)/2
                    corner_chord_r2 = torch.minimum(
                        corner_sides_r2.pow(2).sum(-1).sqrt(),
                        2*r2 )
                    corner_angle_r2 = 2 * ( corner_chord_r2 / (2 * r2) ).arcsin()
                    corner_segm_r2 = r2.pow(2) * (
                        corner_angle_r2 - corner_angle_r2.sin() ) / 2
                    corner_area_r2 = corner_tri_r2 + corner_segm_r2

                    complement_angles_r2 = 2 * (corner_sides_r2 / (2 * r2)).arcsin()
                    complement_area_r2 = (
                        torch.pi * r2.pow(2) - r2.pow(2) * (
                            complement_angles_r2 
                            - complement_angles_r2.sin()
                            ).sum(-1) / 2
                        )
                    area_r2 = ( ext_q1_r2.logical_not() * corner_area_r2
                            + ext_q1_r2 * complement_area_r2 )
                    return (area_r2-area_r1).view(batch_shape)/self.measure() 
                else:
                    """The case when the shell is a circle."""
                    """TODO: Improve efficiency, right now everything is
                    calculated twice, then the unneeded one multiplied by 0,
                    and the two added together at the end."""
                    s = sample.view( ( batch_shape.numel(), self.event_numel ) )
                    c = self.center.flatten()
                    rr = self.radius1.flatten().abs()
                    """If the point is outside the circle in first quadrant,
                    we use a different calculation for the area."""
                    ext_q1_rr = torch.logical_and(
                        (s-c).pow(2).sum(-1) > rr.pow(2),
                        (s > c).prod(-1) )
                    
                    corner_coord_diff_rr = torch.maximum(
                        rr.pow(2) - (c - s).pow(2), 
                        torch.tensor([0,0]) ).sqrt()
                    corner_sides_rr = torch.minimum(
                        corner_coord_diff_rr 
                                    + torch.maximum(
                                        torch.minimum(
                                            (s-c)[:,[1,0]],
                                            corner_coord_diff_rr),
                                        -corner_coord_diff_rr),
                        2*rr  )
                    corner_chord_rr = torch.minimum(
                        corner_sides_rr.pow(2).sum(-1).sqrt(),
                        2*rr )
                    corner_angle_rr = 2 * ( corner_chord_rr / (2 * rr) ).arcsin()
                    corner_arc_rr = corner_angle_rr

                    complement_angles_rr = 2 * (corner_sides_rr / (2 * rr)).arcsin()
                    complement_arc_rr = ( 2 * torch.pi
                                              - complement_angles_rr.sum(-1))
                    arc_rr = ( ext_q1_rr.logical_not() * corner_arc_rr
                            + ext_q1_rr * complement_arc_rr )
                    return arc_rr.view(batch_shape)/(2*torch.pi)
                


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

    

