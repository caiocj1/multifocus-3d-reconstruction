import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from total_variation_3d import TotalVariationL2
from utils.lighting import default_transmittance


class IterTrainer:
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None):
        self.read_config()

        self.obs = input_imgs[0].to(device)
        self.gt = gt_slices
        if self.gt is not None:
            self.gt = self.gt.to(device)
        self.fwd = img_model
        self.layers = input_imgs.shape[-3]

        alpha = default_transmittance(self.flag, self.obs)
        alpha = torch.clip(alpha, min=1./255, max=1)
        alpha = torch.log(alpha).unsqueeze(0).unsqueeze(0) # (1, 1, layer, height, width)

        alpha = alpha.to(device)
        self.omega = alpha.detach().requires_grad_()

        self.optim = torch.optim.Adam([self.omega], lr=self.lr)

        self.loss_fn = nn.MSELoss(reduction="sum")
        self.tv_loss = TotalVariationL2(is_mean_reduction=False)

        if version is not None:
            self.writer = SummaryWriter(log_dir=f"tb_logs/{version}")

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        iter_params = params["IterParams"]

        self.mu = iter_params["mu"]
        self.lr = iter_params["lr"]
        self.n_iter = iter_params["n_iter"]
        self.flag = iter_params["flag"]

    def train(self):
        with tqdm(range(self.n_iter), total=self.n_iter) as pbar:
            for i in pbar:
                # -------- OPTIMIZATION --------
                self.optim.zero_grad()
                loss_sum = 0

                img_list = self.fwd(self.omega)
                for s in range(len(img_list)):
                    out = img_list[s]
                    loss = self.loss_fn(out, self.obs[s])
                    loss.backward()
                    loss_sum = loss_sum + loss.cpu().item()

                constraint = self.mu * self.tv_loss(self.omega)
                constraint.backward()

                loss_sum = loss_sum + constraint.cpu().item()

                self.optim.step()

                # -------- ADD METRICS TO PBAR --------
                if self.gt is not None:
                    trans = torch.exp(self.omega.detach())
                    trans = trans.to('cpu')
                    trans = trans.data.squeeze().float().clamp_(0, 1)
                    psnr = peak_signal_noise_ratio(trans.cpu().detach().numpy(), self.gt[0].cpu().detach().numpy())
                    ssim = structural_similarity(trans.cpu().detach().numpy(), self.gt[0].cpu().detach().numpy(),
                                                 multichannel=False, win_size=self.layers)
                    pbar.set_postfix({"psnr": psnr, "ssim": ssim, "loss": loss_sum})
                else:
                    pbar.set_postfix({"loss": loss_sum})

                # -------- LOG TO TENSORBOARD --------
                if hasattr(self, "writer"):
                    self.writer.add_scalar("loss", loss_sum, global_step=i)
                    if self.gt is not None:
                        self.writer.add_scalar("psnr", psnr, global_step=i)
                        self.writer.add_scalar("ssim", ssim, global_step=i)
