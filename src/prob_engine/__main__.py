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

import click


@click.group()
def cli():
    pass


@cli.command()
def test_uniform():
    from . import uniform_grid
    uniform_grid.test()


@cli.command()
def test_normal():
    from . import multi_normal
    multi_normal.test()


@cli.command()
def test_neural():
    from . import neural_dist
    neural_dist.test()


@cli.command()
def test_mixture():
    from . import mixture
    mixture.test()


@cli.command()
def test_empcdf():
    import torch
    from .uniform_grid import UniformGrid
    grid = UniformGrid(
        torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        torch.tensor([[3, 3], [3, 3]]))

    points = grid.sample(torch.Size((3, 5)))
    print(points.shape)

    values = grid.get_empirical_cdf(100, points)
    print(values.shape)
