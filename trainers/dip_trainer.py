import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.skipnet3d import SkipNet3D
from trainer import Trainer
from utils.pretraining import pretraining_v3, pretraining_sc


class DIPTrainer(Trainer):
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None, weights=None):
        super().__init__(img_model, input_imgs, device, gt_slices, version)

        # ------------- DIP SPECIFIC INIT -------------
        self.weights = weights
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
        dip_params = params["DIPParams"]

        self.n_iter = dip_params["n_iter"]
        self.lr = dip_params["lr"]
        self.pretr_iter = dip_params["pretr_iter"]

    def pretrain(self, type):
        if self.weights is not None:
            print("Both weights and pretraining were given, skipping pretraining and loading weights.")
            return

        writer = self.writer if hasattr(self, "writer") else None
        if type == "v3":
            pretraining_v3(self.inp, self.net, self.pretr_iter, self.version, writer=writer)
        elif type == "sc":
            pretraining_sc(self.inp, self.net, 500, writer=writer)

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
                    psnr, ssim = None, None
                    if self.gt is not None:
                        alpha_np = alpha[0, 0].detach().cpu().numpy()
                        psnr = peak_signal_noise_ratio(alpha_np, self.gt[0].cpu().detach().numpy())
                        ssim = structural_similarity(alpha_np, self.gt[0].cpu().detach().numpy(),
                                                     multichannel=False, win_size=self.layers)
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