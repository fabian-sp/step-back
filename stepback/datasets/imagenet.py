import os
from torchvision import transforms, datasets

input_dim = 224
input_ch = 3
num_classes = 1000


def get_imagenet(split, path):

    if split.lower() not in ('train', 'val'):
        raise ValueError(f"split must be in ('train', 'val'), but got {split}")
    
    # data directory
    traindir = os.path.expanduser(os.path.join(path, "imagenet/train/"))
    valdir = os.path.expanduser(os.path.join(path, "imagenet/val/"))
    
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train = (split.lower() != 'val')
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    dataset = datasets.ImageFolder(
        traindir if train else valdir,
        transform=transform)

    return dataset

