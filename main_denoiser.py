import argparse
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from utils.loading import *
from utils.denoising import real_noise
from models.seeindark import SeeInDark
from trainer_denoiser import DenoisingTrainer

if __name__ == "__main__":
    # ------------------ ARGUMENT PARSING ------------------
    parser = argparse.ArgumentParser(description="Train denoiser", allow_abbrev=False)

    parser.add_argument("--input", "-i", required=True, type=str, help="Path to dataset.")
    parser.add_argument("--version", "-v", type=str, help="version name for Tensorboard")

    args = parser.parse_args()

    if args.version is not None:
        print(args.version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ PREPARE DATASET ------------------
    val_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=0.44531356896770125, std=0.2692461874154524),
    ])

    train_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(),
        transforms.RandAugment(num_ops=2, magnitude=15),
        transforms.ToTensor(),
        #transforms.Normalize(mean=0.44531356896770125, std=0.2692461874154524),
    ])

    train_dataset = ImageFolder(os.path.join(args.input, "train"), transform=train_preprocess)
    # train_dataset.samples = train_dataset.samples[:1000]
    # train_dataset.imgs = train_dataset.imgs[:1000]
    # train_dataset.targets = train_dataset.targets[:1000]
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=512,
                                  num_workers=16,
                                  shuffle=True)

    val_dataset = ImageFolder(os.path.join(args.input, "val"), transform=val_preprocess)
    # val_dataset.samples = val_dataset.samples[:1000]
    # val_dataset.imgs = val_dataset.imgs[:1000]
    # val_dataset.targets = val_dataset.targets[:1000]
    val_dataloader = DataLoader(val_dataset,
                                batch_size=512,
                                num_workers=16,
                                shuffle=False)

    # ------------------ GET MODEL, TRAINER AND TRAIN ------------------
    model = SeeInDark()

    trainer = DenoisingTrainer(model, train_dataloader, val_dataloader, device, version=args.version)
    trainer.train()

