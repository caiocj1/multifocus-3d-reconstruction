# The generic driver requires too many parameters to be provided as input
# this is a modified version, that keeps some of them fixed
# Also, uses one of th precomputed psfs
# input should be the slices dir

import argparse
import yaml
from collections import defaultdict

from trainers.iter_trainer import IterTrainer
from img_model import ImagingModel
from utils.loading import *

if __name__ == "__main__":
    # ------------------ ARGUMENT PARSING ------------------
    parser = argparse.ArgumentParser(description="Run reconstruction", allow_abbrev=False)

    parser.add_argument("--model", "-m", required=True, choices=["iter", "dip", "nf"], help="model selection")
    parser.add_argument("--input", "-i", required=True, help="observed images path")
    parser.add_argument("--ground_truth", "-gt", required=False, help="GT vol path")

    parser.add_argument("--version", "-v", default=None, type=str, help="version name for Tensorboard")
    parser.add_argument("--weights", "-w", default=None, type=str, help="path to model state dict")

    args = parser.parse_args()

    # ------------------ READ CONFIG FILE ------------------
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    main_params = params["MainParams"]

    noise_level = main_params["noise_level"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ GET OBSERVED IMAGES TENSOR ------------------
    in_paths = get_image_paths(args.input)
    img_list = imreads_uint(in_paths, 1)
    imgs = imglist2tensor(img_list)  # tensor(C, N, H, W)

    if noise_level != 0:
        x = imgs.to('cpu').detach().numpy().copy()

        np.random.seed(seed=0)  # for reproducibility
        x += np.random.normal(0, noise_level / 255, x.shape)  # add AWGN

        imgs = torch.from_numpy(x.astype(np.float32)).clone()
    imgs = torch.clip(imgs, min=0, max=1)

    # ------------------ IF GT PATH, GET GT SLICES ------------------
    slices = None
    if args.ground_truth is not None:
        val_paths = get_image_paths(args.ground_truth)
        slice_list = imreads_uint(val_paths, 1)
        slices = imglist2tensor(slice_list)

    # ------------------ ASSIGN MODELS AND TRAINER ------------------
    img_model = ImagingModel(device)

    model_dict = defaultdict()
    model_dict["iter"] = IterTrainer

    trainer = model_dict[args.model](img_model, imgs, device, gt_slices=slices, version=args.version)

    # ------------------ TRAIN ------------------
    trainer.train()
