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

import torch

def get_sample_cdf(event_shape: torch.Size, fixed_sample: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the empirical cumulative distribution function at given 'sample' points,
        using 'fixed_sample' to define the empirical distribution function.
        The shape of 'fixed_sample' should be n + event_shape for n fixed samples,
        and the shape of 'sample' is sample_shape + event_shape.
        The output is of sample_shape.
        """
        sample_shape = sample.shape[:-len(event_shape)]
        fixed_sample_shape = fixed_sample.shape[:-len(event_shape)]
        assert (
            sample.shape == sample_shape + event_shape and
            fixed_sample.shape == fixed_sample_shape + event_shape and
            len(fixed_sample_shape) == 1
            ), "Incorrectly shaped inputs!"
        raw_comp = fixed_sample <= sample.unsqueeze(-len(event_shape)-1)
        for i in range(len(event_shape)):
            raw_comp = raw_comp.all(dim = -1)
        return raw_comp.count_nonzero(dim = -1)/fixed_sample_shape.numel()