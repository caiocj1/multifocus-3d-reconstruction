import torch
import torch.nn as nn
import math
import numpy as np
import os
import yaml


class NeuralField(nn.Module):
    def __init__(self):
        super().__init__()

        self.read_config()

        if self.use_pe:
            self.pe = PositionalEncoding()
            input_size = self.pe.input_size()
        else:
            input_size = 3

        layers1 = [nn.Linear(input_size, self.hidden_size), nn.LeakyReLU()]
        for i in range(((self.n_layers - 2) // 2) - 1):
            layers1.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers1.append(nn.LeakyReLU())
        layers1.append(nn.Linear(self.hidden_size, self.hidden_size))
        layers1.append(nn.LeakyReLU())

        layers2 = [nn.Linear(self.hidden_size + input_size, self.hidden_size), nn.LeakyReLU()]
        for i in range(((self.n_layers - 2) // 2) - 1):
            layers2.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers2.append(nn.LeakyReLU())

        layers2.append(nn.Linear(self.hidden_size, 1))
        self.nf1 = nn.Sequential(*layers1)
        self.nf2 = nn.Sequential(*layers2)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        nf_params = params["NFParams"]

        self.n_layers = nf_params["n_layers"]
        self.hidden_size = nf_params["hidden_size"]
        self.use_pe = nf_params["use_pe"]

    def forward(self, input):
        if hasattr(self, "pe"):
            input = self.pe(input)

        out = self.nf1(input)
        x_concat = torch.concat([input, out], dim=-1)
        out = self.nf2(x_concat)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, l_z = 5, l_xy = 6):
        super().__init__()

        self.l_z = l_z
        self.l_xy = l_xy

        s = torch.sin(torch.arange(0, 180, 45) * np.pi / 180)[:, None]
        c = torch.cos(torch.arange(0, 180, 45) * np.pi / 180)[:, None]
        self.fourier_mapping = torch.cat((s, c), dim=1).T

    def input_size(self):
        return 2 * self.fourier_mapping.shape[1] * self.l_xy + 2 * self.l_z

    def forward(self, pts):
        device = pts.device

        xy_freq = pts[:, 1:] @ torch.tensor(self.fourier_mapping).to(device)

        for l in range(self.l_xy):
            cur_freq = torch.cat([torch.sin(2 ** l * np.pi * xy_freq), torch.cos(2 ** l * np.pi * xy_freq)], dim=-1)
            if l == 0:
                tot_freq = cur_freq
            else:
                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)

        for l in range(self.l_z):
            cur_freq = torch.cat(
                [torch.sin(2 ** l * np.pi * pts[:, 0][:, None]), torch.cos(2 ** l * np.pi * pts[:, 0][:, None])], dim=-1)
            tot_freq = torch.cat([tot_freq, cur_freq], axis=-1)

        return tot_freq
