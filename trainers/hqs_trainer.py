import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.skipnet3d import SkipNet3D
from utils.lighting import default_transmittance
from total_variation_3d import TotalVariationL2
from models.network_dncnn import IRCNN
from trainer import Trainer
from utils.pretraining import pretraining_v3, pretraining_sc, pretraining_v2


class HQSTrainer(Trainer):
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None, weights=None, denoiser_weights=None):
        super().__init__(img_model, input_imgs, device, gt_slices, version)

        # ------------- DIP SPECIFIC INIT -------------
        self.device = device

        alpha = default_transmittance(self.flag, self.obs)
        alpha = torch.clip(alpha, min=1. / 255, max=1)
        alpha = torch.log(alpha).unsqueeze(0).unsqueeze(0)  # (1, 1, layer, height, width)

        alpha = alpha.to(device)
        self.omega = alpha.detach().requires_grad_()

        self.optim = torch.optim.Adam([self.omega], lr=self.lr)

        self.loss_fn = nn.MSELoss(reduction="sum")
        self.error = nn.MSELoss(reduction="sum")

        self.denoiser = IRCNN(in_nc=1, out_nc=1, nc=64).to(device)
        if denoiser_weights is not None:
            state_dict = torch.load(denoiser_weights)
            self.denoiser.load_state_dict(state_dict["0"], strict=True)
        self.denoiser.eval()
        for _, p in self.denoiser.named_parameters():
            p.requires_grad = False

        self.rhos, self.sigmas = self.get_rho_sigma(sigma=max(0.255 / 255., 0),
                                                    iter_num=self.n_iter,
                                                    modelSigma1=2.55,
                                                    modelSigma2=0.255,
                                                    w=1.0)
        self.rhos, self.sigmas = torch.tensor(self.rhos).to(device), torch.tensor(self.sigmas).to(device)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        iter_params = params["HQSParams"]

        self.lr = iter_params["lr"]
        self.n_iter = iter_params["n_iter"]
        self.flag = iter_params["flag"]

    def train(self):
        try:
            former_idx = -1
            with tqdm(range(self.n_iter), total=self.n_iter) as pbar:
                for i in pbar:
                    mu = self.rhos[i].float().item()

                    # -------- DENOISING --------
                    current_idx = np.int32(np.ceil(self.sigmas[i].cpu().numpy() * 255. / 2.) - 1)
                    # print(f'current idx : {current_idx} sigma : {sigmas[i].cpu().numpy()}')
                    if current_idx != former_idx:
                        # model.load_state_dict(model25[str(current_idx)], strict=True)
                        self.denoiser.eval()
                        for _, v in self.denoiser.named_parameters():
                            v.requires_grad = False
                    former_idx = current_idx

                    with torch.no_grad():
                        beta = self.denoiser_3d(self.omega)

                    # -------- OPTIMIZATION --------
                    self.optim.zero_grad()
                    loss_sum = 0

                    img_list = self.fwd(self.omega)
                    for s in range(len(img_list)):
                        out = img_list[s]
                        loss = self.loss_fn(out, self.obs[s])
                        loss.backward()
                        loss_sum = loss_sum + loss.cpu().item()

                    constraint = 0.5 * mu * self.error(self.omega, beta)
                    constraint.backward()

                    loss_sum = loss_sum + constraint.cpu().item()

                    self.optim.step()

                    # -------- ADD METRICS TO PBAR --------
                    psnr, ssim = None, None
                    if self.gt is not None:
                        trans = torch.exp(self.omega.detach())
                        trans = trans.to('cpu')
                        trans = trans.data.squeeze().float().clamp_(0, 1)
                        psnr = peak_signal_noise_ratio(trans.cpu().detach().numpy(), self.gt[0].cpu().detach().numpy())
                        ssim = structural_similarity(trans.cpu().detach().numpy(), self.gt[0].cpu().detach().numpy(),
                                                     channel_axis=False, win_size=self.layers)
                        pbar.set_postfix({"psnr": psnr, "ssim": ssim, "loss": loss_sum})
                    else:
                        pbar.set_postfix({"loss": loss_sum})

                    # -------- LOG TO TENSORBOARD --------
                    self.log_metrics(i, loss_sum, psnr, ssim)
                    vol_list = [self.obs, torch.stack(img_list), self.gt, torch.exp(self.omega)] \
                        if self.gt is not None else \
                        [self.obs, torch.stack(img_list), torch.exp(self.omega)]
                    vol_list = [vol.cpu().detach().numpy() for vol in vol_list]
                    self.log_figs(i, *vol_list)
        except KeyboardInterrupt:
            print("Training interrupted.")

        self.log_hparams()
        np.save(f"tb_logs/{self.version}/alpha.npy", self.omega.cpu().detach().numpy())

    def denoiser_3d(self, alpha):
        beta = torch.squeeze(alpha)
        beta = torch.unsqueeze(beta, dim=1)

        # --------------------------------
        # denoiser in xy plane
        # --------------------------------
        xy_slice = beta
        xy_slice = self.denoiser(xy_slice)

        # --------------------------------
        # denoiser in yz plane
        # --------------------------------
        yz_slice = beta.permute(3, 1, 0, 2)  # yz平面
        yz_slice = self.denoiser(yz_slice)
        yz_slice = yz_slice.permute(2, 1, 3, 0)

        # --------------------------------
        # denoiser in zx plane
        # --------------------------------
        zx_slice = beta.permute(2, 1, 3, 0)  # zx平面
        zx_slice = self.denoiser(zx_slice)
        zx_slice = zx_slice.permute(3, 1, 0, 2)
        # print(zx_slice)
        beta = (xy_slice + yz_slice + zx_slice) / 3
        beta = torch.squeeze(beta)
        beta = torch.unsqueeze(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=0)

        return beta

    def get_rho_sigma(self, sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
        '''
        One can change the sigma to implicitly change the trade-off parameter
        between fidelity term and prior term
        '''
        modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
        modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
        sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
        rhos = list(map(lambda x: 1.0*(sigma**2)/(x**2), sigmas))
        return rhos, sigmas