import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from total_variation_3d import TotalVariationL2
from utils.lighting import default_transmittance
from models.skipnet3d import SkipNet3D

class DIPTrainer:
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None, weights=None):
        self.read_config()

        self.obs = input_imgs[0].to(device)
        self.gt = gt_slices
        if self.gt is not None:
            self.gt = self.gt.to(device)
        self.fwd = img_model
        self.layers = input_imgs.shape[-3]

        if version is not None:
            self.writer = SummaryWriter(log_dir=f"tb_logs/{version}")

        # ------------- DIP SPECIFIC INIT -------------
        self.net = SkipNet3D().to(device)

        if weights is not None:
            self.net.load_state_dict(torch.load(weights + "/net.pt"))
            self.inp = torch.load(weights + "/inp.pt")
        else:
            cell_dim = tuple(input_imgs.shape[-3:])
            self.inp = self.inp = torch.zeros((1, 1) + cell_dim, device=device, dtype=torch.float32)
            self.inp.uniform_()
            self.inp *= 1.0 / 10  # inp_noise_var
            self.inp = self.inp.detach().clone()

        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.loss_fn = nn.L1Loss()

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        iter_params = params["DIPParams"]

        self.n_iter = iter_params["n_iter"]
        self.lr = iter_params["lr"]

    def train(self):
        try:
            with tqdm(range(self.n_iter), total=self.n_iter) as pbar:
                for i in pbar:
                    # -------- OPTIMIZATION --------
                    self.optim.zero_grad()

                    alpha = self.net(self.inp)
                    out = self.fwd(torch.log(alpha))
                    out = torch.stack(out)

                    loss = self.loss_fn(out, self.obs)
                    loss.backward()
                    self.optim.step()

                    # -------- ADD METRICS TO PBAR --------
                    if self.gt is not None:
                        alpha_np = alpha[0, 0].detach().cpu().numpy()
                        psnr = peak_signal_noise_ratio(alpha_np, self.gt[0].cpu().detach().numpy())
                        ssim = structural_similarity(alpha_np, self.gt[0].cpu().detach().numpy(),
                                                     multichannel=False, win_size=self.layers)
                        pbar.set_postfix({"psnr": psnr, "ssim": ssim, "loss": loss.item()})
                    else:
                        pbar.set_postfix({"loss": loss.item()})

                    # -------- LOG TO TENSORBOARD --------
                    if hasattr(self, "writer"):
                        self.writer.add_scalar("loss", loss.item(), global_step=i)
                        if self.gt is not None:
                            self.writer.add_scalar("psnr", psnr, global_step=i)
                            self.writer.add_scalar("ssim", ssim, global_step=i)
        except KeyboardInterrupt:
            print("Training interrupted.")

        if hasattr(self, "writer"):
            img_model_hparams = {key: vars(self.fwd)[key] for key in vars(self.fwd)
                                 if (type(vars(self.fwd)[key]) == int or type(vars(self.fwd)[key]) == float)}
            self.writer.add_text("img_model_hparams", str(img_model_hparams), 0)

            self.writer.flush()
            self.writer.close()