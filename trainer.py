from torch.utils.tensorboard import SummaryWriter

from utils.plotting import plot_slices


class Trainer:
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None):
        self.read_config()

        self.obs = input_imgs[0].to(device)
        self.gt = gt_slices
        if self.gt is not None:
            self.gt = self.gt.to(device)
        self.fwd = img_model
        self.layers = input_imgs.shape[-3]
        self.version = version

        if version is not None:
            self.writer = SummaryWriter(log_dir=f"tb_logs/{version}")

    def read_config(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def log_metrics(self, step, loss, psnr=None, ssim=None):
        if hasattr(self, "writer"):
            self.writer.add_scalar("loss", loss, global_step=step)
            if psnr is not None:
                self.writer.add_scalar("psnr", psnr, global_step=step)
            if ssim is not None:
                self.writer.add_scalar("ssim", ssim, global_step=step)

    def log_figs(self, step, *vols):
        vol_list = list(vols)
        if hasattr(self, "writer") and step % 100 == 0:
            for i in range(len(vol_list)):
                if len(vol_list[i].shape) == 5:
                    vol_list[i] = vol_list[i][0, 0]
                elif len(vol_list[i].shape) == 4:
                    vol_list[i] = vol_list[i][0]

            fig = plot_slices(*vol_list)
            self.writer.add_figure("training/img", fig, global_step=step)

    def log_hparams(self):
        trainer_params = {key: vars(self)[key] for key in vars(self)
                          if (type(vars(self)[key]) == int or type(vars(self)[key]) == float)}
        self.writer.add_text("trainer_hparams", str(trainer_params), 0)

        img_model_hparams = {key: vars(self.fwd)[key] for key in vars(self.fwd)
                             if (type(vars(self.fwd)[key]) == int or type(vars(self.fwd)[key]) == float)}
        self.writer.add_text("img_model_hparams", str(img_model_hparams), 0)

        if hasattr(self, "net"):
            net_params = {key: vars(self.net)[key] for key in vars(self.net)
                          if (type(vars(self.net)[key]) == int or type(vars(self.net)[key]) == float)}
            self.writer.add_text("trainer_hparams", str(net_params), 0)

        self.writer.flush()
        self.writer.close()
