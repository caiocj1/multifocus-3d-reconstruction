from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

from utils.plotting import plot_slices


class PretrainingDataset(Dataset):
    def __init__(self, nsamples, shape):
        super().__init__()
        self.data = self.generate_data(nsamples, shape)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def generate_data(self, nsamples, shape):
        dataset = []

        for i in range(nsamples):
            n_ellipsoids = np.random.randint(low=4, high=12)
            Y = np.zeros(shape[-3:])
            for j in range(n_ellipsoids):
                a, b, c = np.random.uniform(low=1, high=8, size=3)
                z = np.linspace(-10, 10, shape[-3]) + np.random.uniform(low=-8, high=8)
                x = np.linspace(-10, 10, shape[-2]) + np.random.uniform(low=-8, high=8)
                y = np.linspace(-10, 10, shape[-1]) + np.random.uniform(low=-8, high=8)

                ellipsoid = (x / a) ** 2 + (y[:, None] / b) ** 2 + (z[:, None, None] / c) ** 2 <= 1

                alpha = np.random.uniform(low=0.05, high=0.1)
                ellipsoid = ellipsoid.astype(float) * alpha
                Y += ellipsoid

            Y = np.clip(Y, 0, 1)
            X = Y.copy() + np.random.uniform(high=0.2, size=Y.shape)
            X = np.clip(X, 0, 1)
            dataset.append((1 - X, 1 - Y))

        return dataset


def pretraining_v3(inp, net, n_iter, version, writer=None):
    target_shape = list(inp.shape)

    device = next(net.parameters()).device

    x, y = PretrainingDataset(1, target_shape)[0]
    y = torch.tensor(y[None, None]).to(device).float()

    optim = torch.optim.Adam(net.parameters())
    lossfn = torch.nn.MSELoss()

    try:
        with tqdm(range(n_iter), total=n_iter) as pbar:
            for i in pbar:
                alpha = net(inp)

                loss = lossfn(alpha, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

                if writer is not None:
                    writer.add_scalar("pretraining_loss", loss.item(), global_step=i)
                    if i % 10 == 0:
                        fig = plot_slices(y[0, 0].cpu().detach().numpy(),
                                          alpha[0, 0].cpu().detach().numpy())
                        writer.add_figure(f"pretraining_train/img", fig, global_step=i)

    except KeyboardInterrupt:
        print('Received keyboard interrupt. Stopping pre-training.')

    torch.save(net.state_dict(), f"tb_logs/{version}/net.pt")
    torch.save(inp, f"tb_logs/{version}/inp.pt")


def pretraining_sc(inp, net, n_iter, cval=0.75, writer=None):
    target_shape = list(inp.shape)
    target_shape[1] = net.out_channels
    # modify nch.

    target = cval * torch.ones(target_shape, device=inp.device)
    # separate optimizer because this should be treated independently
    # Also the target is different
    optim = torch.optim.Adam(net.parameters())
    lossfn = torch.nn.MSELoss()
    # just trying to fit a constant value, MSE should be enough

    try:
        with tqdm(range(n_iter), total=n_iter) as pbar:
            for i in pbar:
                alpha = net(inp)

                # just trying to fit a constant value, MSE should be enough
                loss = lossfn(alpha, target)

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

                if writer is not None:
                    writer.add_scalar("pretraining_loss", loss.item(), global_step=i)

    except KeyboardInterrupt:
        print('Received keyboard interrupt. Stopping pre-training.')


def pretraining_const(inp, net, niter=100, cval=0.75):
    """
    Neural field fit to constant value as initialization
    :return: None
    """
    target = cval * torch.ones((len(inp), 1), device=inp.device)
    optim = torch.optim.Adam(net.parameters())
    lossfn = torch.nn.MSELoss()
    # just trying to fit a constant value, MSE should be enough

    try:
        with tqdm(range(niter + 1)) as pbar:
            for i in pbar:
                alpha = net(inp)

                # just trying to fit a constant value, MSE should be enough
                loss = lossfn(alpha, target)

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(loss='{:.10f}'.format(loss.item()))

    except KeyboardInterrupt:
        print('Received keyboard interrupt. Stopping pre-training.')
