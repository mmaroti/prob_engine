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

import abc
import torch
from typing import Iterator


class Distribution(abc.ABC):
    def __init__(self, sample_shape: torch.Size):
        """
        Creates an abstract multi variate distribution whose samples have
        the specified shape.
        """
        assert isinstance(sample_shape, torch.Size)
        self._sample_shape = sample_shape

    @property
    def sample_shape(self) -> torch.Size:
        """
        Returns the shape of samples that this distribution can produce.
        """
        return self._sample_shape

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns the list of parameters of this parametric distribution.
        """
        if False:
            yield

    @abc.abstractmethod
    def sample(self, batch_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Randomly samples from the distribution batch many times and returns
        a tensor of shape batch_shape + sample_shape.
        """
        raise NotImplemented()
