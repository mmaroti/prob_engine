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

def df_func(func: Callable[[torch.Tensor], torch.Tensor],
            order: int, lower_orders: bool
            )-> Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor], tuple[torch.Tensor, ...]]:
    """
    Returns the 'order'-differentiated 'func' function.
    Currently uses torch.func.jacrev for the sake of performance,
    consider replacing with torch.autograd.functional.jacobian
    in the future when it becomes stable and efficient.
    """
    assert order >= 0
    if order == 0:
        result = func
    elif order == 1:
        result = torch.func.jacrev(
            func, 0, has_aux=lower_orders)
    elif order > 1 and (not lower_orders):
        temp = func
        for i in range(0,order):
            temp = torch.func.jacrev(
                temp, 0, has_aux=False)
        result = temp
    elif order > 1 and lower_orders:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    return result

def get_df_func(func: Callable[[torch.Tensor], torch.Tensor],
           order: int, input: torch.Tensor)->torch.Tensor:
    """
    Evaluates the 'order'-differentiated 'func'
    function on 'input' point.
    Assumes that inputs are not batched.
    """
    assert order >= 0
    if order == 0:
        result = func(input)
    else:
        if input.requires_grad is False:
            input.requires_grad_(True)
        df = df_func(func, order, False)
        result = df(input)
    return result

def get_df_func_batched(func: Callable[[torch.Tensor], torch.Tensor],
           order: int, input: torch.Tensor)->torch.Tensor:
    """
    Evaluates the 'order'-differentiated 'func'
    function on 'input' points.
    Assumes that 'input' is batched
    in its first dimension.
    """
    assert order >= 0
    if order == 0:
        result = func(input)
    else:
        if input.requires_grad is False:
            input.requires_grad_(True)
        df = df_func(func, order, False)
        result = torch.vmap(df, 0)(input)
    return result

def get_df1(fx: torch.Tensor, x: torch.Tensor,
           graph: bool) -> torch.Tensor:
    """
    Returns the (first-order) jacobian
    of 'fx' with respect to 'x'.
    Assumes that inputs are not batched.
    """
    assert x.requires_grad is True
    if flat_fx.numel() == 1:
        result = torch.autograd.grad(fx, x, 
                create_graph=graph,
                allow_unused=True)[0]
    else:
        x_shape = x.shape
        fx_shape = fx.shape
        flat_fx = fx.flatten()
        result = []
        for i in range(flat_fx.numel()):
            select = torch.tensor(
                [0]*i + [1] + [0]*(flat_fx.numel()-i-1))
            fxi = (flat_fx*select).sum()
            result.append(
                torch.autograd.grad(fxi, x,
                        create_graph=graph,
                        allow_unused=True))[0]
        result = torch.stack(result, 0)
    return result.view(fx_shape + x_shape)

def get_df(fx: torch.Tensor, x: torch.Tensor,
           order: int, graph: bool) -> torch.Tensor:
    """
    Returns the 'order'-rank jacobian
    of 'fx' with respect to 'x'.
    Assumes that inputs are not batched.
    """
    assert order >= 0
    assert x.requires_grad is True
    if order == 0:
        result = fx
    elif order == 1:
        result = get_df1(fx, x, graph)
    else:
        x_shape = x.shape
        fx_shape = fx.shape
        result = fx
        for i in range(order):
            if i < order-1:
                result = get_df1(result, x, True)
            else:
                result = get_df1(result, x, graph)
    return result.view(fx_shape + x_shape*order)

def get_df_batched(fx: torch.Tensor, x: torch.Tensor,
           order: int, graph: bool) -> torch.Tensor:
    """
    Returns the 'order'-rank "jacobian"
    of 'fx' with respect to 'x'.
    Assumes that both inputs are batched,
    with the same batch shape, the first dimension.
    """
    assert x.requires_grad is True
    assert x.dim() > 1 and fx.dim() > 1
    assert x.shape[0] == fx.shape[0]
    def temp_df(tempfx: torch.Tensor, tempx: torch.Tensor):
        return get_df(tempfx, tempx, order, graph)
    result = torch.vmap(temp_df, in_dims = 0, out_dims=0)(fx, x)
    return result