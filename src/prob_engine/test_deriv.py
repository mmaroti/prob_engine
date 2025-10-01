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

from typing import Callable
import torch


def fun(xs: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.prod(xs) + torch.sum(xs)


def test():
    xs = torch.Tensor([1.0, 2.0, 3.0]).requires_grad_(True)
    y = fun(xs)
    print(y)

    d0, = torch.autograd.grad(y, xs, create_graph=True, materialize_grads=True)
    print(d0[0])

    d1, = torch.autograd.grad(
        d0[0], xs, create_graph=True, materialize_grads=True)
    print(d1[1])

    d2, = torch.autograd.grad(
        d1[1], xs, create_graph=True, materialize_grads=True)
    print(d2[2])
