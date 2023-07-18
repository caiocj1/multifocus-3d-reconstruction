import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.plotting import plot_multifigures
from utils.denoising import real_noise


class DenoisingTrainer:
    def __init__(self, denoiser, train_dataloader, val_dataloader, device, version=None):
        self.read_config()

        self.model = nn.DataParallel(denoiser.to(device))
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.version = version
        self.device = device

        self.n_iter = 100
        self.loss_fn = nn.L1Loss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.n_iter * len(self.train_dataloader))
        self.normalize = transforms.Normalize(mean=0.44531356896770125, std=0.2692461874154524)

        if version is not None:
            self.writer = SummaryWriter(log_dir=f"tb_logs/{version}")

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        self.denoising_params = params["DenoiserParams"]

    def train_loop(self, epoch):
        self.model.train()
        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                  desc=f"Train Epoch {epoch}", leave=False) as pbar:
            total_loss = 0.0

            for i, batch in pbar:
                x, _ = batch
                x = x.to(self.device)
                x_noise = x + torch.tensor(real_noise(x.cpu().numpy(), **self.denoising_params)).to(self.device)
                inp = self.normalize(x_noise)
                inp = inp.detach().clone()

                out = self.model(inp.float())

                loss = self.loss_fn(out, x)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.lr_scheduler.step()

                # --------------------- LOGGING ---------------------
                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

                self.writer.add_scalar("lr", self.optim.param_groups[0]['lr'],
                                       global_step=epoch * len(self.train_dataloader) + i)

                psnr = peak_signal_noise_ratio(out.cpu().detach().numpy(), x.cpu().detach().numpy())
                self.writer.add_scalar("psnr/train_step", psnr, global_step=i)

                ssim = structural_similarity(out.cpu().detach().numpy(), x.cpu().detach().numpy(), channel_axis=1)
                self.writer.add_scalar("ssim/train_step", ssim, global_step=i)

                total_loss += loss.item()
                self.writer.add_scalar("loss/train_step", loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)

                if i % 100 == 0:
                    fig = plot_multifigures(x_noise, out, x)
                    self.writer.add_figure("denoising/training", fig, global_step=epoch * len(self.train_dataloader) + i)
                    plt.close(fig)

        epoch_loss = total_loss / len(self.train_dataloader)

        return epoch_loss

    def val_loop(self, epoch):
        self.model.eval()
        with torch.no_grad():
            with tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader),
                      desc=f"Val Epoch {epoch}", leave=False) as pbar:
                total_loss = 0.0
                avg_psnr = 0.0
                avg_ssim = 0.0

                for i, batch in pbar:
                    x, _ = batch
                    x = x.to(self.device)
                    x_noise = x + torch.tensor(real_noise(x.cpu().numpy(), **self.denoising_params)).to(self.device)
                    inp = self.normalize(x_noise)
                    inp = inp.detach().clone()

                    out = self.model(inp.float())

                    loss = self.loss_fn(out, x)

                    # --------------------- LOGGING ---------------------
                    pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

                    avg_psnr += peak_signal_noise_ratio(out.cpu().detach().numpy(), x.cpu().detach().numpy())
                    avg_ssim += structural_similarity(out.cpu().detach().numpy(), x.cpu().detach().numpy(), channel_axis=1)

                    total_loss += loss.item()

                    if i % 50 == 0:
                        fig = plot_multifigures(x_noise, out, x)
                        self.writer.add_figure(f"denoising/val_step_{i}", fig, global_step=epoch * len(self.val_dataloader) + i)
                        plt.close(fig)

        epoch_loss = total_loss / len(self.val_dataloader)
        avg_psnr /= len(self.val_dataloader)
        avg_ssim /= len(self.val_dataloader)

        self.writer.add_scalar("psnr/val_epoch", avg_psnr, global_step=epoch)
        self.writer.add_scalar("ssim/val_epoch", avg_ssim, global_step=epoch)

        return epoch_loss

    def train(self):
        try:
            for epoch in range(self.n_iter):
                epoch_loss = self.train_loop(epoch)
                self.writer.add_scalar("loss/train_epoch", epoch_loss, global_step=epoch)

                epoch_loss = self.val_loop(epoch)
                self.writer.add_scalar("loss/val_epoch", epoch_loss, global_step=epoch)

            print("Training ended.")
        except KeyboardInterrupt:
            print("Training interrupted.")

        torch.save(self.model.state_dict(), f"tb_logs/{self.version}/denoiser_model.pt")
        self.log_hparams()

    def log_hparams(self):
        self.writer.add_text("noise_params", str(self.denoising_params), 0)

        self.writer.flush()
        self.writer.close()
