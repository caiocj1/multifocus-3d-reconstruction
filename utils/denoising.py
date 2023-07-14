import numpy as np
from scipy.stats import tukeylambda


def real_noise(img, gain, shape, color_bias, sigma_tl, sigma_r):
    noise_p = np.random.poisson(lam=img * 255) / 255 - img
    noise_read = tukeylambda.rvs(shape, loc=color_bias, scale=sigma_tl, size=img.shape)
    noise_r = np.random.normal(0, sigma_r, size=(*img.shape[:-1], 1))
    noise_q = np.random.uniform(low=-1 / (2 * 255), high=1 / (2 * 255), size=img.shape)
    return gain * noise_p + noise_read + noise_r + noise_q
