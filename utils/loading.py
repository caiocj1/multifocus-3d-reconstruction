"""
All credits for this file go to the authors of the repositories:
https://github.com/cszn
https://github.com/twhui/SRGAN-pyTorch
https://github.com/xinntao/BasicSR
"""

import os
import cv2
import torch
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# ----------------------------------------------------------------------
def get_image_paths(dir_root):
    paths = None  # return None if dair_root is None
    if dir_root is not None:
        paths = sorted(_get_paths_from_images(dir_root))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

# ----------------------------------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def imreads_uint(paths, n_channels=3):
    #  input: paths
    # output: [HxWx3(RGB or GGG)], or [HxWx1 (G)]
    imgs = []
    for idx, path in enumerate(paths):
        img = imread_uint(path, n_channels)
        imgs.append(img)
    return imgs

# ----------------------------------------------------------------------
def imglist2tensor(img_list):
    for idx, img in enumerate(img_list):
        if idx == 0:
            imgs = np.copy(np.expand_dims(img, axis=0))
        else:
            imgs = np.concatenate([imgs, np.expand_dims(img, axis=0)])
    imgs = imgs / 255.0
    imgs = torch.from_numpy(imgs)
    imgs = imgs.to(torch.float32)
    imgs = imgs.permute(3, 0, 1, 2)  # NHWC -> CNHW

    return imgs

# ----------------------------------------------------------------------
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
