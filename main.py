import argparse
import yaml
from collections import defaultdict

from trainers.iter_trainer import IterTrainer
from trainers.dip_trainer import DIPTrainer
from trainers.nf_trainer import NFTrainer
from img_model import ImagingModel
from utils.loading import *


if __name__ == "__main__":
    # ------------------ ARGUMENT PARSING ------------------
    parser = argparse.ArgumentParser(description="Run reconstruction", allow_abbrev=False)

    parser.add_argument("--model", "-m", required=True, choices=["iter", "dip", "nf"], help="model selection")
    parser.add_argument("--input", "-i", required=True, help="observed images path")

    parser.add_argument("--ground_truth", "-gt", help="GT vol path")
    parser.add_argument("--pretraining", "-p", choices=["sc", "v2", "v3", "const"], help="type of pretraining")
    parser.add_argument("--version", "-v", type=str, help="version name for Tensorboard")
    parser.add_argument("--weights", "-w", type=str, help="path to load model state dict")
    parser.add_argument("--noise_level", "-n", type=float, default=0, help="variance of noise to apply to observations")
    parser.add_argument("--psf_mask", "-psf", type=str, help="path to psf mask .npy")

    args = parser.parse_args()

    if args.version is not None:
        print(args.version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ GET OBSERVED IMAGES TENSOR ------------------
    in_paths = get_image_paths(args.input)
    img_list = imreads_uint(in_paths, 1)
    imgs = imglist2tensor(img_list)  # tensor(C, N, H, W)

    if args.noise_level != 0:
        x = imgs.to('cpu').detach().numpy().copy()

        np.random.seed(seed=0)  # for reproducibility
        x += np.random.normal(0, args.noise_level, x.shape)  # add AWGN

        imgs = torch.from_numpy(x.astype(np.float32)).clone()
    imgs = torch.clip(imgs, min=0, max=1)

    # ------------------ IF GT PATH, GET GT SLICES ------------------
    slices = None
    if args.ground_truth is not None:
        val_paths = get_image_paths(args.ground_truth)
        slice_list = imreads_uint(val_paths, 1)
        slices = imglist2tensor(slice_list)

    # ------------------ ASSIGN MODELS AND TRAINER ------------------
    img_model = ImagingModel(device, args.psf_mask)

    model_dict = defaultdict()
    model_dict["iter"] = IterTrainer
    model_dict["dip"] = DIPTrainer
    model_dict["nf"] = NFTrainer

    trainer = model_dict[args.model](img_model, imgs, device,
                                     gt_slices=slices, version=args.version, weights=args.weights)

    # ------------------ TRAIN ------------------
    if args.pretraining is not None:
        trainer.pretrain(args.pretraining)
    trainer.train()
