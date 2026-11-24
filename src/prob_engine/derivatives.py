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

import torch

def get_partial(
        x: torch.Tensor, fx: torch.Tensor,
        coords: list[int]) -> torch.Tensor:
    """
    Returns the mixed partial derivatives of 'fx'
    as a function of 'x', according to 'coords'.
    'x' is not necessarily a vector.
    If coords=[0,1], returns d_1(d_2(f)), where
    d_i is partial derivation by the i-th argument.
    Assumes 'x' and 'fx' are not batched.
    """
    if len(coords) == 0:
        result = fx
    else:
        assert min(coords) >= 0
        assert max(coords) < x.numel()
        assert x.requires_grad == True
        result = fx
        for c in coords:
            result = result.flatten()
            temp = []
            for i in range(result.numel()):
                temp.append(
                    torch.autograd.grad(
                        result[i], x,
                        create_graph=True,
                        allow_unused=True)[0]
                    )
            result = torch.stack(temp, 0).view(
                fx.numel(), x.numel()
            )[:, c].view(fx.shape)
    return result

def get_partial_batched(
        x: torch.Tensor, fx: torch.Tensor,
        coords: list[int]) -> torch.Tensor:
    """
    Returns the mixed partial derivatives of 'fx'
    as a function of 'x', according to 'coords'.
    'x' is not necessarily a vector.
    If coords=[0,1], returns d_1(d_2(f)), where
    d_i is partial derivation by the i-th argument.
    Assumes that both 'x' and 'fx'
    are batched along the 0-th dimension.
    """
    if len(coords) == 0:
        result = fx
    else:
        assert min(coords) >= 0
        assert max(coords) < x.shape[1:].numel()
        assert x.requires_grad == True
        assert x.shape[0]==fx.shape[0]
        fx_dim = fx.shape[1:].numel()
        result = fx
        for c in coords:
            result = result.sum(0).flatten()
            temp = []
            for i in range(result.numel()):
                temp.append(
                    torch.autograd.grad(
                        result[i], x,
                        create_graph=True,
                        allow_unused=True)[0]
                    )
            result = torch.stack(temp, -1).view(
                x.shape[0], x.shape[1:].numel(), fx_dim
            )[:, c, :].view(fx.shape)
    return result