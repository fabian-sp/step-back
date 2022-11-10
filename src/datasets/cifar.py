import torchvision
from torchvision import transforms


def get_cifar10(split, path): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == 'train':
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
            
        ds = torchvision.datasets.CIFAR10(root=path, train=True, 
                                          download=True,
                                          transform=transform_train
                                          )
    elif split == 'val':
        
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            normalize,
                                            ])
        
        ds = torchvision.datasets.CIFAR10(root=path, train=False, 
                                          download=True, 
                                          transform=transform_val
                                          )
        
    return ds