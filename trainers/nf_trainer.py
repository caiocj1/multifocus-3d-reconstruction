import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.pretraining import pretraining_const
from models.neural_field import NeuralField
from trainer import Trainer


class NFTrainer(Trainer):
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None, weights=None):
        super().__init__(img_model, input_imgs, device, gt_slices, version)

        # ------------- NF SPECIFIC INIT -------------
        self.net = NeuralField().to(device)

        self.cell_dim = tuple(input_imgs.shape[-3:])
        n_z, n_x, n_y = self.cell_dim
        x = torch.linspace(-1, 1, n_x).to(device) / 10
        y = torch.linspace(-1, 1, n_y).to(device) / 10
        z = torch.linspace(-1, 1, n_z).to(device) / 10

        self.inp = torch.cartesian_prod(z, x, y).to(device)

        self.loss_fn = nn.L1Loss()

        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, [500], gamma=0.5)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        nf_params = params["NFParams"]

        self.n_iter = nf_params["n_iter"]
        self.lr = nf_params["lr"]

    def pretrain(self, type):
        if type == "const":
            pretraining_const(self.inp, self.net)
        else:
            raise Exception("Invalid pretraining version.")

    def train(self):
        try:
            with tqdm(range(self.n_iter), total=self.n_iter) as pbar:
                for i in pbar:
                    # -------- OPTIMIZATION --------
                    self.optim.zero_grad()

                    alpha = self.net(self.inp).reshape(self.cell_dim)[None, None]
                    out = self.fwd(torch.log(alpha))
                    out = torch.stack(out)

                    loss = self.loss_fn(out, self.obs)

                    loss.backward()
                    self.optim.step()

                    # self.scheduler.step()

                    # -------- ADD METRICS TO PBAR --------
                    psnr, ssim = None, None
                    if self.gt is not None:
                        alpha_np = np.clip(alpha[0, 0].detach().cpu().numpy(), 0, 1)
                        psnr = peak_signal_noise_ratio(alpha_np, self.gt[0].cpu().detach().numpy())
                        ssim = structural_similarity(alpha_np, self.gt[0].cpu().detach().numpy(),
                                                     channel_axis=False, win_size=self.layers)
                        pbar.set_postfix({"psnr": psnr, "ssim": ssim, "loss": loss.item()})
                    else:
                        pbar.set_postfix({"loss": loss.item()})

                    # -------- LOG TO TENSORBOARD --------
                    self.log_metrics(i, loss.item(), psnr, ssim)
                    vol_list = [self.obs, out, self.gt, alpha] if self.gt is not None else [self.obs, out, alpha]
                    vol_list = [vol.cpu().detach().numpy() for vol in vol_list]
                    self.log_figs(i, *vol_list)
        except KeyboardInterrupt:
            print("Training interrupted.")

        self.log_hparams()
