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

from prob_engine.distribution import Distribution


class UniformBall(Distribution):
    def __init__(self,
                 center: torch.Tensor,
                 radius: torch.Tensor,
                 device: Optional[str] = None):
        
        assert center.dim() == 1 and radius.numel() == 1
        center = center.flatten()
        radius = radius.flatten()
        Distribution.__init__(self, center.shape[-1], device=device)
        self._center = torch.nn.Parameter(
            center.to(dtype=torch.float32, device=self._device))
        self._radius = torch.nn.Parameter(radius.abs().flatten().to(
            dtype=torch.float32, device=self._device))

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
        if self._event_size == 1:
            return self._radius.abs() * 2
        elif self._event_size == 2:
            return self._radius.pow(2) * torch.pi
        elif self._event_size == 3:
            return self._radius.abs().pow(3) * 3 * torch.pi / 4
        else:
            c1 = torch.tensor(torch.pi, device=self._device
                              ).pow(self._event_size/2.0)
            c2 = torch.tensor(1.0 + self._event_size/2.0,
                              device=self._device).lgamma().exp()
            return self._radius.abs().pow(self._event_size) * c1 / c2

    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        directions = torch.normal(mean=0.0, std=1.0,
                                  size=batch_shape + (self._event_size,),
                                  device=self._device)
        directions /= directions.pow(2).sum(-1).sqrt().unsqueeze(-1)
        rad = torch.rand(size=batch_shape, device=self._device
                         ).pow(1.0/self._event_size) * self._radius.abs()
        return self._center + (directions * rad.unsqueeze(-1)
                               ).view(batch_shape + self.event_shape)

    def get_pdf(self, sample: torch.Tensor) -> torch.Tensor:
        assert (0 < self._radius.abs()).all(), "No density function exists!"
        batch_shape = sample.shape[:-1]
        assert sample.shape == batch_shape + self.event_shape
        sample = sample.view(batch_shape + (self._event_size,)
                             ).to(device=self._device)
        inside = (sample - self._center.flatten()
                  ).pow(2).sum(-1) <= self._radius.pow(2)
        return inside/self.measure()

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.log(self.get_pdf(sample))

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-1]
        assert sample.shape == batch_shape + self.event_shape
        sample = sample.to(device=self._device)
        if (self._radius.item() == 0):
            probs = (sample >= self._center).view(
                batch_shape + (self._event_size, )
            ).all(-1)
            return probs
        else:
            if self._event_size == 1:
                probs = torch.minimum(
                    (sample - self._center + self._radius.abs()).relu()
                    / self.measure(),
                    torch.tensor(1.0, device=self._device))
                return probs.view(batch_shape)

            elif self._event_size == 2:
                sam = sample.view((batch_shape.numel(), self._event_size))
                cen = self._center.flatten()
                rad = self._radius.abs().flatten()
                """If the point is outside the circle in first quadrant,
                we use a different calculation for the area."""
                c = (sam >= cen + rad).all(-1)
                b = torch.logical_and(
                    (sam > cen).all(-1).logical_and(c.logical_not()),
                    (sam-cen).pow(2).sum(-1) > rad.pow(2))
                a = b.logical_or(c).logical_not()

                diffs = (rad.pow(2)-(cen - sam).pow(2)).relu().sqrt()
                sides = torch.minimum(
                    diffs + torch.maximum(
                        torch.minimum((sam-cen)[:, [1, 0]], diffs),
                        -diffs),  2*rad)
                chord_a = torch.minimum(sides[a].pow(2).sum(-1).sqrt(), 2*rad)
                angle_a = 2 * (chord_a / (2 * rad)).arcsin()
                area_a = sides[a].prod(-1)/2 + \
                    rad.pow(2) * (angle_a - angle_a.sin()) / 2
                angles_b = 2 * (sides[b] / (2 * rad)).arcsin()
                area_b = torch.pi - (angles_b - angles_b.sin()).sum(-1) / 2
                area_b = rad.pow(2) * area_b
                area = torch.zeros(batch_shape.numel(), device=self._device)
                area[c] = self.measure()
                area[a] = area_a
                area[b] = area_b
                return area.view(batch_shape)/self.measure()
            else:
                raise NotImplementedError()


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
