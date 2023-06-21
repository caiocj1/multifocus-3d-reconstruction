from utils.lighting import *

import torch
import torch.nn.functional as F
import yaml
import os

class ImagingModel:
    def __init__(self, device):
        self.read_config()

        diameter = set_diameter(self.NA, self.layers, self.z_res)
        self.ray_num, ray_check = set_incidental_light(diameter, self.apa_size)
        self.intensity = 1 / self.ray_num

        self.ray_mat = range_matrix_generation(self.ray_num, ray_check, self.layers,
                                               diameter, self.apa_size, self.xy_res, self.z_res)
        self.ray_mat = torch.from_numpy(self.ray_mat).clone()  # (ray_num, 2*layer-1, , )
        self.ray_mat = self.ray_mat.to(torch.float32).to(device)

        self.padding_size = int((self.ray_mat.size()[2] - 1) / 2)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        img_params = params["ImgModelParams"]

        self.apa_size = img_params["apa_size"]
        self.xy_res = img_params["xy_res"]
        self.z_res = img_params["z_res"]
        self.NA = img_params["NA"]
        self.layers = img_params["layers"]

    def __call__(self, omega):
        img_list = []
        for s in range(self.layers):
            out = F.conv2d(omega[0],
                           self.ray_mat[:, self.layers - 1 - s:2 * self.layers - 1 - s, :, :],
                           padding=self.padding_size)
            out = self.intensity * torch.sum(torch.exp(out), dim=1).squeeze()
            img_list.append(out)
        return img_list
