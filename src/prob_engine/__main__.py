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
    from . import normal
    normal.test()


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


@cli.command()
def test_discrete():
    from . import discrete
    discrete.test()

@cli.command()
def test_mixture_normal():
    from . import mixture_normal
    mixture_normal.test()


@cli.command()
def test_uniform_grid_ball():
    from .testers import uniform_grid_ball
    uniform_grid_ball.test()

@cli.command()
def test_uniform_grid_shell():
    from .testers import uniform_grid_shell
    uniform_grid_shell.test()

@cli.command()
def test_uniform_grid_normal():
    from .testers import uniform_grid_normal
    uniform_grid_normal.test()

@cli.command()
def NN_training_test_1():
    from .training.NN_to_uniform_grid import NN_train_test_1
    NN_train_test_1()

@cli.command()
def NN_training_test_2():
    from .training.NN_to_uniform_grid import NN_train_test_2
    NN_train_test_2()

@cli.command()
def test_deriv():
    from .test_deriv import test
    test()

@cli.command()
def test_ugrid_composition():
    from .composed_grids_tests import test1, test2, test3, test4
    test1(); test2(); test3(); test4()

@cli.command()
def test_einsum_cdf():
    from . import einsum_attempts
    einsum_attempts.einsum_test()
    einsum_attempts.einsum_comp_test()