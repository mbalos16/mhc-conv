import torch
import argparse

# Model Dependencies
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup

# ===Import the data ===
from data_loader import load_cifar, load_imagenet

# === Import the model ===
from mhc import mHCResNet

# === Import the trainer
from trainer import train


# ==================== Define Argparse for Comand Line Training ====================
AVAILABLE_DATASETS = ["cifar100","imagenet"]


def parse_args():
    parser = argparse.ArgumentParser(
        description = "A script that helps run mHC architecture for convolutional models.\
        An example on how to run the model would be:\
        `python main.py --dataset cifar100`" 
        )

    parser.add_argument(
        "-d",
        "--dataset", 
        required = True,
        type = str, 
        help = f"The name of the dataset to be used for training.\
        Available datasets: {AVAILABLE_DATASETS}"
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type = int,
        required = False,
        default = 2500,
        help = "Number of epochs to train the model."
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type = float,
        required = False,
        default = 0.001,
        help = "The learning rate to be used for the optimizer."
    )

    parser.add_argument(
        "-wp",
        "--warmup-period",
        type = int,
        required = False,
        default = 2000,
        help = "The warmup period needed to warm up the learning rate. \
        Makes hyperparameter tuning more robust while improving the final performance."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.dataset not in AVAILABLE_DATASETS:
        raise ValueError(
            "The dataset required for training: {args.dataset} is not available. \
            Please select one of the following options: {AVAILABLE_DATASETS}."
        )
    if args.dataset == "cifar100":
        dataloaders, num_classes = load_cifar()
        print(f"Cifar 100 dataloaders have loaded. The number of classes is {num_classes}.")
    elif args.dataset =="imagenet":
        dataloaders, num_classes = load_imagenet()
        print(f"Imagenet dataloaders have loaded. The number of classes is {num_classes}.")

    model = mHCResNet(num_outputs = num_classes)
    summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_period)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tensor Board Logs Class
    writer = SummaryWriter()

    train(
        epochs = args.epochs, 
        dataloaders = dataloaders, 
        optimizer = optimizer,
        device = device,
        model = model,
        writer = writer,
        warmup_scheduler = warmup_scheduler,
    )

if __name__ =="__main__":
    main()