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

import torch, math
import matplotlib as plt
import torchvision.transforms as tf
from PIL import Image
from prob_engine.uniform_grid import UniformGrid

def get_uniformgrid_from_image(path: str = "", counts: torch.Tensor = torch.tensor([512,512])) -> UniformGrid:
    if path == "":
        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        img_path = dir_path+"\\images\\handdrawn.png"
        file_path = img_path
    else:
        file_path = path
    assert counts.dim() == 1 and counts.numel() == 2
    image = Image.open(file_path)
    my_tf = tf.Compose([
        tf.PILToTensor(),
        tf.Grayscale(),
    ])
    img_tensor = my_tf(image)
    assert img_tensor.dim() == 3 and img_tensor.shape[0] == 1
    img_tensor = img_tensor.squeeze(0)
    ug = UniformGrid(torch.tensor([[0.0,0.0],[1.0,1.0]]), counts)
    ug._parameter = torch.nn.Parameter(img_tensor.to(dtype = torch.float32))
    return ug

def test():
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = dir_path+"\\images\\handdrawn.png"
    image = Image.open(img_path)
    ug = get_uniformgrid_from_image()
    image.show()
    ug.plot_exact_pdf(0.0, 1.0, bins = min(ug._counts.max().item(), 512))
    image.close()
    ug.plot_exact_cdf(0.0, 1.0, bins = min(ug._counts.max().item(), 64))
