from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from functools import partial
import os

# Transform the labels into one_hot_encoding.
def one_hot_encoding_labels(label, num_classes):
    one_hot_encoding_labels = torch.nn.functional.one_hot(
        torch.tensor(label),
        num_classes=num_classes,
    )
    return one_hot_encoding_labels


def load_cifar(batch_size_train = 8, batch_size_validation = 8, num_classes = 100):
    """
    A function that helps load the cifar100 dataset and transform it to create the dataloader for model training.
    """

    # ========== Train ========== 
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ]
    )
    
    train_dataset = CIFAR100(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform_train, 
        target_transform = partial(one_hot_encoding_labels, num_classes = num_classes)
    )

    
    # ========== Validation ========== 
    transform_validation = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
            
        ]
    )
    validation_dataset = CIFAR100(
        root = "./data",
        train = False,
        download = True, 
        transform = transform_validation,
        target_transform = partial(one_hot_encoding_labels, num_classes = num_classes)
    )

    
    # ========== Dataloader ========== 
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True),
        "val": DataLoader(validation_dataset, batch_size = batch_size_validation, shuffle = False)
                  }

    return dataloaders, num_classes



def load_imagenet(batch_size_train = 8, batch_size_validation = 8, num_classes =1000):
    global_data_path = "/mnt/FLASH/imagenet"
    train_path_img = os.path.join(global_data_path, "train")
    validation_path_img = os.path.join(global_data_path, "validation")
    
    # ========== Train ==========
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    train_dataset = ImageFolder(
        root=train_path_img,  # Path to the dataset root directory
        transform=transform_train,  # Apply the defined transformations
        target_transform=partial(one_hot_encoding_labels, num_classes=num_classes),
    )
    
    # ========== Validation ==========
    transform_val = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    validation_dataset = ImageFolder(
        root=validation_path_img,  # Path to the dataset root directory
        transform=transform_val,  # Apply the defined transformations
        target_transform=partial(one_hot_encoding_labels, num_classes=num_classes),
    )
    
    # ========== Dataloader ==========
    
    # A dictionary is defined to easily switch between training and validation datasets during training.
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=8, shuffle=True),
        "val": DataLoader(validation_dataset, batch_size=8, shuffle=False),
    }
    return dataloaders, num_classes

