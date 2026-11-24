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

import math
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tf
from PIL import Image
from pathlib import Path


class cdf_layer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int,
                  resolution: int, proj_dim: int, device=None):
        super().__init__()
        assert size_in > 0 and size_out > 0 and resolution > 0 and proj_dim > 0
        assert size_in == 2     # Temporarily restricted to 2 dimensions.
        assert proj_dim == size_in    # Temporarily restricted.
        self.size_in = size_in
        self.size_out = size_out
        self.resolution = resolution    # Could be changed to [res_x, res_y, ...].

        if device is None:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device = "cuda:" + str(device_count-1) # Use the last GPU.
            else:
                device = "cpu"
        else:
            assert isinstance(device, torch.device)
        
        print(device)
        self._device = device
        
        proj_count = torch.combinations(torch.range(0,size_in), r = proj_dim,
                                        with_replacement=False).shape[0]
        proj_count = size_in    # Temporarily restricted.

        self.weight = torch.nn.Parameter(
            torch.empty(
                #[self.size_out, proj_count] + [self.resolution]*proj_dim,
                [self.resolution]*2 + [self.size_out],
                device = self._device, dtype = torch.float32),
            requires_grad = True)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Randomizes the weights.
        """
        torch.nn.init.uniform_(self.weight, 0.0, 1.0)
    
    def set_weights(self, new_weights: torch.Tensor):
        """
        Manually set the weights.
        """
        n_w = new_weights.to(device=self._device, dtype=torch.float32)
        assert new_weights.shape == self.weight.shape
        self.weight = torch.nn.Parameter(n_w.abs(), requires_grad = True)

    def weight_measure(self):
        """
        Returns the total weights.
        Temporary calculation method, to be changed later.
        """
        w = self.weight.abs()
        result = w.sum(0).sum(0)
        print(w.shape, result.shape)
        return result
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_shape = input.shape[:-1]
        assert input.shape == batch_shape + (self.size_in,)
        C = torch.zeros((self.resolution+1, self.resolution+1, self.size_out),
                        device=self._device, dtype=self.weight.dtype)
        w_m = self.weight_measure()
        w = self.weight.abs()/(w_m[None,None,:])
        C[1:, 1:, :] = w.abs().cumsum(0).cumsum(1)

        pts = input.clamp(0.0,1.0)\
            .view((batch_shape.numel(), self.size_in))\
            .to(dtype=torch.float32, device=self._device)\
             * self.resolution
        
        x, y = pts[:, 0], pts[:, 1]
        ix, iy = torch.floor(x).long(), torch.floor(y).long()
        fx, fy = x - ix, y - iy

        ix1 = (ix + 1).clamp(max=self.resolution)
        iy1 = (iy + 1).clamp(max=self.resolution)

        result = ((1 - fx) * (1 - fy)).unsqueeze(-1) * C[ix, iy,:]\
                    + (fx * (1 - fy)).unsqueeze(-1) * C[ix1, iy,:]\
                    + ((1 - fx) * fy).unsqueeze(-1) * C[ix, iy1,:]\
                    + (fx * fy).unsqueeze(-1) * C[ix1, iy1,:]

        return result.view(batch_shape + (self.size_out,))

def get_tensor_for_image(path: str = "", counts: torch.Tensor = torch.tensor([32,32])) -> torch.Tensor:
    if path == "":
        cur_dir = Path(__file__).resolve().parent
        file_path = cur_dir / "images" / "handdrawn.png"
    else:
        file_path = path
    assert counts.dim() == 1 and counts.numel() == 2
    image = Image.open(file_path)
    my_tf = tf.Compose([
        tf.PILToTensor(),
        tf.Grayscale(),
        tf.Resize(counts.tolist()),
    ])
    img_tensor = my_tf(image)
    assert img_tensor.dim() == 3 and img_tensor.shape[0] == 1
    img_tensor = img_tensor.squeeze(0)
    return img_tensor

def get_partial(
        x: torch.Tensor, fx: torch.Tensor,
        coords: list[int]) -> torch.Tensor:
    """
    Returns the mixed partial derivatives of 'fx'
    as a function of 'x', according to 'coords'.
    'x' is not necessarily a vector.
    If coords=[0,1], returns d_2(d_1(f)), where
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
    If coords=[0,1], returns d_2(d_1(f)), where
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

def plot_net(net: torch.nn.Module, bins: int = 60, device = None):
    if device is None:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device = "cuda:" + str(device_count-1) # Use the last GPU.
            else:
                device = "cpu"
    else:
        assert isinstance(device, torch.device)
    width = 1.0 / bins
    sample1 = torch.linspace(
        0.0 + 0.5 * width,
        1.0 - 0.5 * width,
        bins,
        dtype=torch.float32,
        device=device)
    sample2 = torch.meshgrid([sample1, sample1],
                             indexing="xy")
    sample2_flat = torch.stack(sample2, dim=-1)\
                .view(torch.Size((bins*bins,2)))\
                .to(device=device)\
                .requires_grad_(True)
    vals = net.forward(sample2_flat)

    plt.pcolormesh(
        sample1.cpu().detach().numpy(),
        sample1.cpu().detach().numpy(),
        vals.view((bins,bins)).cpu().detach().numpy(),
        rasterized=True)
    plt.colorbar()
    plt.title("Plot of net CDF")
    plt.show()
    deriv = get_partial_batched(sample2_flat, vals, [0,1])
    plt.pcolormesh(
        sample1.cpu().detach().numpy(),
        sample1.cpu().detach().numpy(),
        deriv.view((bins,bins)).cpu().detach().numpy(),
        rasterized=True)
    plt.colorbar()
    plt.title("Plot of net PDF")
    plt.show()

def test_basic():
    print("Testing basic functionality.")
    test_layer = cdf_layer(2, 2, 2, 2)
    print("Weight shape", test_layer.weight.shape, "Weight measure", test_layer.weight_measure())
    points = torch.tensor([[0.0,0.0],[0.0,0.2],[0.2,0.2],[1.0,0.2],[0.2,1.0],[0.8,1.0],[0.8,0.8],[1.0,1.0]],
                          requires_grad=True)
    print("Weights", test_layer.weight)
    fs = test_layer.forward(points)
    print("Evaluation", fs)
    derivs = get_partial_batched(points, fs, [0,1])
    print("Derivatives", derivs)

def test_image():
    print("Testing image reconstruction.")
    img_t = get_tensor_for_image()
    img_net = cdf_layer(2, 1, img_t.shape[-1], 2)
    old_w = img_net.weight
    img_net.set_weights(
        img_t.view(old_w.shape).to(dtype=old_w.dtype,
                                    device = old_w.device))
    plot_net(img_net)

def test_plotting():
    print("Testing plotting of cdf and corresponding pdf.")
    test_layer = cdf_layer(2, 1, 2, 2)
    plot_net(test_layer)
    test_layer2 = torch.nn.Sequential(
        cdf_layer(2, 2, 5, 2),
        cdf_layer(2, 2, 5, 2),
        cdf_layer(2, 1, 5, 2),
    )
    plot_net(test_layer2)

def cdf_net_tests():
    test_basic()
    test_plotting()
    test_image()

cdf_net_tests()