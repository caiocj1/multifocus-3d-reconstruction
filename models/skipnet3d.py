# same skipnet with 3d convs

import torch
from torch import nn
import os
import yaml

class SkipNet3D(nn.Module):
    # skipnet from the code of dip

    def __init__(self):
        super().__init__()

        self.read_config()

        self.down = nn.ModuleList([])
        self.down.append(
            Down(self.in_channels, self.down_channels[0], self.filter_size_down, self.pad))
        for i in range(len(self.down_channels) - 1):
            self.down.append(
                Down(self.down_channels[i], self.down_channels[i + 1], self.filter_size_down, self.pad))

        self.up = nn.ModuleList([])

        for i in range(0, len(self.up_channels) - 1, 1):
            self.up.append(
                UpConcat(
                    self.up_channels[i + 1] + self.skip_channels[i],
                    self.up_channels[i],
                    self.filter_size_up,
                    self.pad,
                    self.upsample_mode,
                ))
        self.up.append(
            UpConcat(
                self.down_channels[-1] + self.skip_channels[-1],
                self.up_channels[-1],
                self.filter_size_up,
                self.pad,
                self.upsample_mode,
            ))

        self.skip = nn.ModuleList([])
        self.skip.append(
            ConvBlock(self.in_channels, self.skip_channels[0], self.filter_size_skip, pad=self.pad))
        for i in range(1, len(self.skip_channels)):
            self.skip.append(
                ConvBlock(self.down_channels[i-1],
                          self.skip_channels[i],
                          self.filter_size_skip,
                          pad=self.pad))

        self.final_conv = nn.Conv3d(self.up_channels[0],
                                    self.out_channels,
                                    kernel_size=1,
                                    padding=0)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        skipnet_params = params["DIPParams"]

        self.in_channels = skipnet_params["in_channels"]
        self.out_channels = skipnet_params["out_channels"]
        self.nscale = skipnet_params["nscale"]
        self.down_channels = [skipnet_params["up_down_channels"]] * self.nscale
        self.up_channels = [skipnet_params["up_down_channels"]] * self.nscale
        self.skip_channels = [skipnet_params["skip_channels"]] * self.nscale
        self.pad = skipnet_params["pad"]
        self.upsample_mode = skipnet_params["upsample_mode"]

        self.filter_size_down = skipnet_params["filter_size_down"]
        self.filter_size_up = skipnet_params["filter_size_up"]
        self.filter_size_skip = skipnet_params["filter_size_skip"]

    def forward(self, x):
        x_s = []
        x_out = x

        for i in range(len(self.down)):
            x_s.append(self.skip[i](x_out))
            x_out = self.down[i](x_out)
            # print(x_out.shape)

        # up is stored in the reverse order
        for i in range(len(self.up) - 1, -1, -1):
            x_out = self.up[i](x_out, x_s[i])
            # print(x_out.shape)

        out = torch.sigmoid(self.final_conv(x_out))
        # print(out.shape)

        return out


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pad):
        super().__init__()

        mid_channels = out_channels

        layers = []
        layers.append(
            ConvBlock(in_channels,
                      mid_channels,
                      kernel_size,
                      stride=2,
                      pad=pad))
        layers.append(
            ConvBlock(mid_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      pad=pad))

        self.conv_down = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_down(x)


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad="zero"):

        super().__init__()

        if pad != "zero" and pad != "reflection":
            raise TypeError(f"Unknown padding type {pad}")

        to_pad = int((kernel_size - 1) / 2)

        layers = []
        if pad == "reflection":
            layers.append(nn.ReflectionPad3d(to_pad))
            to_pad = 0

        layers.append(
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=to_pad))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.convblock(x)


class UpConcat(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pad, mode):

        super().__init__()

        mid_channels = out_channels

        self.up = nn.Upsample(scale_factor=2, mode=mode)

        layers = []

        layers.append(nn.BatchNorm3d(in_channels))
        layers.append(
            ConvBlock(in_channels, mid_channels, kernel_size, pad=pad))

        # 1x1 up
        layers.append(
            ConvBlock(mid_channels, out_channels, kernel_size=1, pad=pad))

        self.op = nn.Sequential(*layers)

    def forward(self, x1, x2):
        # x1 -> from down
        # x2 -> skip connection
        # make x1 like x2 after upsample
        # x1 likely to be same or higher in size than x2

        x1 = self.up(x1)

        # target shape
        _, _, x2d, x2h, x2w = x2.shape

        diffZ = (x1.size()[2] - x2.size()[2]) // 2
        diffY = (x1.size()[3] - x2.size()[3]) // 2
        diffX = (x1.size()[4] - x2.size()[4]) // 2

        # print(x1.shape, x2.shape)
        # the diffrence in size is expected to be at most 1.
        # the diffs will be 0, always.
        assert(diffX == 0)
        assert(diffY == 0)
        assert(diffZ == 0)

        # x1 -> NCDHW
        x1 = x1[:, :, diffZ:x2d + diffZ, diffY:x2h + diffY, diffX:x2w + diffX]

        # print(x1.shape, x2.shape)
        x = torch.cat([x1, x2], dim=1)

        return self.op(x)


# if __name__ == "__main__":
#     # testing code
#     in_channels = 32
#     height = 50
#     width = 50
#     depth = 11
#
#     out_channels = 1
#     nscale = 3
#     down_channels = [128] * nscale
#     up_channels = [128] * nscale
#     skip_channels = [4] * nscale
#     pad = "reflection"
#     upsample_mode = "trilinear"
#
#     net = SkipNet3D(
#         in_channels,
#         out_channels,
#         down_channels,
#         up_channels,
#         skip_channels,
#         pad=pad,
#         upsample_mode=upsample_mode,
#     )
#
#     rand_inp = torch.rand((1, in_channels, depth, height, width),
#                           dtype=torch.float32)
#
#     out = net(rand_inp)
#     print(out.shape)
