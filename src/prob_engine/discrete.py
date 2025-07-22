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


class DiscreteUniform(Distribution):
    def __init__(self,
                 points: torch.Tensor,
                 device: Optional[str] = None):
        assert points.dim() > 1 and points.shape[0] > 0

        Distribution.__init__(self, points.shape[1:], device=device)
        self._atoms = Parameter(points.to(dtype=torch.float32, device=self._device))
    
    @property
    def atom_num(self) -> int:
        return self._atoms.shape[0]

    @property
    def bounds(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return torch.stack((mins, maxs), 0)

    @property
    def min_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        return mins

    @property
    def max_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return maxs

    @property
    def parameters(self) -> Iterator[Parameter]:
        yield self._atoms

    def initialize(self, atoms: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):] 
        assert atoms.shape == batch_shape + self._event_shape
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        self._atoms = Parameter(atoms)
    
    def add_atoms(self, new_atoms: torch.Tensor):
        if new_atoms.shape == self._event_shape:
            new_atoms = new_atoms.unsqueeze(0).to(
                device=self._device)
            self._atoms = Parameter(
                torch.cat((self._atoms, new_atoms), 0))
        else:
            batch_shape = new_atoms.shape[:-len(self._event_shape)]
            assert new_atoms.shape == batch_shape + self._event_shape
            new_atoms = new_atoms.view(
                (batch_shape.numel(),) + self._event_shape
                ).to(device=self._device)
            self._atoms = Parameter(
                torch.cat((self._atoms, new_atoms), 0))
    
    def delete_atoms(self, delete_points: torch.Tensor):
        raise NotImplementedError()
    
    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        selection = torch.randint(0, self.atom_num, batch_shape,
                                   device=self._device)
        return self._atoms[selection].view(batch_shape + self._event_shape)

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view((batch_shape.numel(), 1, self.event_numel)
                             ).to(dtype=torch.float32, device=self._device)
        atoms = self._atoms.view(
            torch.Size((self.atom_num, self.event_numel)))
        result = (atoms <= sample).all(-1).count_nonzero(-1).to(dtype=torch.float32)
        result *= (1.0/self.atom_num)
        return result.view(batch_shape)
    
class DiscreteFixedPoints(Distribution):
    def __init__(self,
                 points: torch.Tensor,
                 device: Optional[str] = None):
        assert points.dim() > 1 and points.shape[0] > 0

        Distribution.__init__(self, points.shape[1:], device=device)
        self._atoms = Parameter(points.to(dtype=torch.float32, device=self._device))
        self._weights = Parameter(torch.rand(size=[self._atoms.shape[0]],
                                            dtype=torch.float32, device=self._device))    
    @property
    def atom_num(self) -> int:
        return self._atoms.shape[0]

    @property
    def bounds(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return torch.stack((mins, maxs), 0)

    @property
    def min_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        return mins

    @property
    def max_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return maxs
    
    @property
    def atoms(self) -> torch.Tensor:
        return self._atoms
    
    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def parameters(self) -> Iterator[Parameter]:
        yield self._weights
    
    def reset_weights(self):
        weights = torch.rand(self._weights.shape,
                    dtype=torch.float32, device=self._device)
        self._weights = Parameter(weights)    
    
    def initialize_points(self, atoms: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):]
        assert atoms.shape == batch_shape + self._event_shape
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        weights = torch.rand(size=[len(atoms.shape[0])],
                            dtype=torch.float32, device=self._device)
        self._atoms = Parameter(atoms)
        self._weights = Parameter(weights)
    
    def initialize_weights(self, weights: torch.Tensor):
        assert weights.numel() == self.atom_num
        assert weights.count_nonzero() > 0
        weights = weights.view((self.atom_num,)).to(
            dtype=torch.float32, device=self._device)
        self._weights = Parameter(weights)


    def initialize(self, atoms: torch.Tensor, weights: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):]
        assert atoms.shape == batch_shape + self._event_shape
        assert weights.numel() == batch_shape.numel()
        assert weights.count_nonzero() > 0
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        weights = weights.view((batch_shape.numel(),)).to(
            dtype=torch.float32, device=self._device)
        self._atoms = Parameter(atoms)
        self._weights = Parameter(weights)
    
    def add_atoms(self, new_atoms: torch.Tensor):
        raise NotImplementedError()
    
    def delete_atoms(self, delete_points: torch.Tensor):
        raise NotImplementedError()
    
    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        selection = torch.multinomial(
            self._weights.abs(),
            batch_shape.numel(),
            replacement=True)
        selected = self._atoms[selection].to(device=self._device)
        return selected.view(batch_shape + self._event_shape)

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view((batch_shape.numel(), 1, self.event_numel)
                             ).to(dtype=torch.float32, device=self._device)
        atoms = self._atoms.view(
            torch.Size((self.atom_num, self.event_numel)))
        result = (atoms <= sample).all(-1).to(dtype=torch.float32)
        w = self._weights.abs()
        w *= 1.0/w.sum()
        result = (result * w).sum(-1)
        return result.view(batch_shape)

class DiscreteFixedWeights(Distribution):
    def __init__(self,
                 points: torch.Tensor,
                 device: Optional[str] = None):
        assert points.dim() > 1 and points.shape[0] > 0

        Distribution.__init__(self, points.shape[1:], device=device)
        self._atoms = Parameter(points.to(dtype=torch.float32, device=self._device))
        self._weights = Parameter(torch.rand(size=[self._atoms.shape[0]],
                                            dtype=torch.float32, device=self._device))    
    @property
    def atom_num(self) -> int:
        return self._atoms.shape[0]

    @property
    def bounds(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return torch.stack((mins, maxs), 0)

    @property
    def min_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        return mins

    @property
    def max_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return maxs
    
    @property
    def atoms(self) -> torch.Tensor:
        return self._atoms
    
    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def parameters(self) -> Iterator[Parameter]:
        yield self._atoms

    def reset_weights(self):
        self._weights = Parameter(torch.rand(size=[self.atom_num],
                             dtype=torch.float32, device=self._device))    
    
    def initialize_points(self, atoms: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):]
        assert atoms.shape == batch_shape + self._event_shape
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        weights = torch.rand(size=[len(atoms.shape[0])],
                            dtype=torch.float32, device=self._device)
        self._atoms = Parameter(atoms)
        self._weights = Parameter(weights)
    
    def initialize_weights(self, weights: torch.Tensor):
        assert weights.numel() == self.atom_num
        assert weights.count_nonzero() > 0
        weights = weights.view((self.atom_num,)).to(
            dtype=torch.float32, device=self._device)
        self._weights = Parameter(weights)


    def initialize(self, atoms: torch.Tensor, weights: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):]
        assert atoms.shape == batch_shape + self._event_shape
        assert weights.numel() == batch_shape.numel()
        assert weights.count_nonzero() > 0
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        weights = weights.view((batch_shape.numel(),)).to(
            dtype=torch.float32, device=self._device)
        self._atoms = Parameter(atoms)
        self._weights = Parameter(weights)
    
    def add_atoms(self, new_atoms: torch.Tensor):
        raise NotImplementedError()
    
    def delete_atoms(self, delete_points: torch.Tensor):
        raise NotImplementedError()
    
    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        selection = torch.multinomial(
            self._weights.abs(),
            batch_shape.numel(),
            replacement=True
        )
        selected = self._atoms[selection].to(device=self._device)
        return selected.view(batch_shape + self._event_shape)

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view((batch_shape.numel(), 1, self.event_numel)
                             ).to(dtype=torch.float32, device=self._device)
        atoms = self._atoms.view(
            torch.Size((self.atom_num, self.event_numel)))
        result = (atoms <= sample).all(-1).to(dtype=torch.float32)
        w = self._weights.abs()
        w *= 1.0/w.sum()
        result = (result * w).sum(-1)
        return result.view(batch_shape)

class Discrete(Distribution):
    def __init__(self,
                 points: torch.Tensor,
                 device: Optional[str] = None):
        assert points.dim() > 1 and points.shape[0] > 0

        Distribution.__init__(self, points.shape[1:], device=device)
        self._atoms = Parameter(points.to(dtype=torch.float32, device=self._device))
        self._weights = Parameter(torch.rand(size=[self._atoms.shape[0]],
                                            dtype=torch.float32, device=self._device))    
    @property
    def atom_num(self) -> int:
        return self._atoms.shape[0]

    @property
    def bounds(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return torch.stack((mins, maxs), 0)

    @property
    def min_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        mins = torch.min(shaped_atoms,0).values.view(self.event_shape)
        return mins

    @property
    def max_bound(self) -> torch.Tensor:
        shaped_atoms = self._atoms.view((self.atom_num, self.event_numel))
        maxs = torch.max(shaped_atoms,0).values.view(self.event_shape)
        return maxs
    
    @property
    def atoms(self) -> torch.Tensor:
        return self._atoms
    
    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def parameters(self) -> Iterator[Parameter]:
        yield self._atoms
        yield self._weights
    
    def reset_weights(self):
        self._weights = Parameter(torch.rand(size=[self.atom_num],
                             dtype=torch.float32, device=self._device))    
    
    def initialize_points(self, atoms: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):]
        assert atoms.shape == batch_shape + self._event_shape
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        weights = torch.rand(size=[len(atoms.shape[0])],
                            dtype=torch.float32, device=self._device)
        self._atoms = Parameter(atoms)
        self._weights = Parameter(weights)
    
    def initialize_weights(self, weights: torch.Tensor):
        assert weights.numel() == self.atom_num
        assert weights.count_nonzero() > 0
        weights = weights.view((self.atom_num,)).to(
            dtype=torch.float32, device=self._device)
        self._weights = Parameter(weights)


    def initialize(self, atoms: torch.Tensor, weights: torch.Tensor):
        batch_shape = atoms.shape[-len(self._event_shape):]
        assert atoms.shape == batch_shape + self._event_shape
        assert weights.numel() == batch_shape.numel()
        assert weights.count_nonzero() > 0
        atoms = atoms.view((batch_shape.numel(),) + self._event_shape
                             ).to(dtype=torch.float32,
                                  device=self._device)
        weights = weights.view((batch_shape.numel(),)).to(
            dtype=torch.float32, device=self._device)
        self._atoms = Parameter(atoms)
        self._weights = Parameter(weights)
    
    def add_atoms(self, new_atoms: torch.Tensor):
        raise NotImplementedError()
    
    def delete_atoms(self, delete_points: torch.Tensor):
        raise NotImplementedError()
    
    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        selection = torch.multinomial(
            self._weights.abs(),
            batch_shape.numel(),
            replacement=True
        )
        selected = self._atoms[selection].to(device=self._device)
        return selected.view(batch_shape + self._event_shape)

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        batch_shape = sample.shape[:-len(self._event_shape)]
        assert sample.shape == batch_shape + self._event_shape
        sample = sample.view((batch_shape.numel(), 1, self.event_numel)
                             ).to(dtype=torch.float32, device=self._device)
        atoms = self._atoms.view(
            torch.Size((self.atom_num, self.event_numel)))
        result = (atoms <= sample).all(-1).to(dtype=torch.float32)
        w = self._weights.abs()
        w *= 1.0/w.sum()
        result = (result * w).sum(-1)
        return result.view(batch_shape)

def test():
    disc = Discrete(torch.tensor([[0,0],[0,1],[1,0],[1,1],[1.0/3,0.5],[2.0/3,0.2]]))
    print("Parameters", list(disc.parameters))
    disc.plot_empirical_pdf()
    disc.plot_empirical_cdf()
    disc.plot_exact_cdf()
    
    disc_U = DiscreteUniform(torch.tensor([[0,0],[0,1],[1,0],[1,1],[1.0/3,0.5],[2.0/3,0.2]]))
    print("Parameters", list(disc_U.parameters))
    disc_U.plot_exact_cdf()
    disc_U.add_atoms(torch.tensor([[-1,0],[-1,-1],[0,-1]]))
    disc_U.plot_exact_cdf()
    print("Parameters", list(disc_U.parameters))

