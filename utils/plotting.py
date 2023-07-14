import matplotlib.pyplot as plt


def plot_slices(*args):
    shape = args[0].shape

    nrows = len(args)
    fig, axs = plt.subplots(nrows, shape[0])

    row = 0
    for sample in args:
        for i in range(shape[0]):
            axs[row, i].imshow(sample[i], cmap="gray", vmin=0, vmax=1)
            axs[row, i].axis("off")
        row += 1

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    height = (20 / shape[0]) * nrows
    fig.set_size_inches(20, height)
    return fig

def plot_figure(array):
    fig = plt.figure()
    plt.imshow(array, cmap="gray")
    plt.axis("off")
    return fig

def plot_multifigures(*imgs):
    shape = imgs[0].shape

    n_figs = len(imgs)
    fig, axs = plt.subplots(1, n_figs)
    for i in range(n_figs):
        axs[i].imshow(imgs[i][0, 0].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
        axs[i].axis("off")

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    width = 6 * n_figs
    fig.set_size_inches(width, 6)
    return fig
