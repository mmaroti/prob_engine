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
from torch.nn import Sequential
from prob_engine.distribution import Distribution
from prob_engine.discrete import Discrete
from prob_engine.mixture_normal import MixtureNormal
from prob_engine.training.training_misc import train_distribution, cdf_cdf_Lp, cdf_cdf_E_Lp, multilinspace
from prob_engine.neural_dist import *
from prob_engine.testers.uniform_grid_normal import UniformGridNormal

class test_nn_1(Distribution):
    def __init__(self, event_size: int, device: Optional[str] = None):

        Distribution.__init__(self, event_size, device=device)
        self.model = TensorOutputLayer(Sequential(
            PosLinearLayer(event_size,10*event_size),
            Relu2Layer(),
            PosLinearLayer(20*event_size,20*event_size),
            Relu2Layer(),
            PosLinearLayer(40*event_size, 10),
            Relu2Layer(),
            PosLinearLayer(20*event_size,1),
        ))

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model.forward(sample).view(sample.shape[:-1])

class test_nn_1(Distribution):
    def __init__(self, event_size: int, device: Optional[str] = None):

        Distribution.__init__(self, event_size, device=device)
        self.model = TensorOutputLayer(Sequential(
            PosLinearLayer(event_size,10*event_size),
            Relu2Layer(),
            PosLinearLayer(20*event_size,20*event_size),
            Relu2Layer(),
            UniformMixLayer(20),
            UniformMixLayer(10),
            PosConvexLayer(40*event_size, 20*event_size),
            PosLinearLayer(20*event_size, 1),
            UniformMixLayer(10),
            ClampLayer(),
        ))

    @property
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def get_cdf(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model.forward(sample).view(sample.shape[:-1])
    

def NN_train_test_1():
    model_distribution = test_nn_1(1)
    target_distribution = UniformGridNormal(torch.tensor([0.0]), torch.tensor([0.2]),
                                            torch.tensor([[-1.0],[1.0]]), torch.tensor([100]), True)
    def my_loss(dist: Distribution,
                target: Distribution) -> torch.Tensor:
        return cdf_cdf_E_Lp(dist, target,torch.tensor([[-1],[1]]), 10, 1000, 2.0)
    print("CDF", model_distribution.get_cdf(sample = torch.tensor([0.0])))
    train_distribution(model_distribution, target_distribution, steps = 10001, loss_func = my_loss)

def NN_train_test_2():
    model_distribution = test_nn_1(2)
    target_distribution = UniformGridNormal(torch.tensor([0.0,0.0]), torch.tensor([0.2, 0.4]),
                                            torch.tensor([[-1.0,-1.0],[1.0,1.0]]),
                                            torch.tensor([100,100]), True)
    def my_loss(dist: Distribution,
                target: Distribution) -> torch.Tensor:
        return cdf_cdf_E_Lp(dist, target,torch.tensor([[-1,-1],[1,1]]), 10, 500, 2.0)
    print("CDF", model_distribution.get_cdf(sample = torch.tensor([0.0,0.0])))
    train_distribution(model_distribution, target_distribution, steps = 2001, loss_func = my_loss)