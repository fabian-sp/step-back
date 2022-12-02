"""
Normalization values are from:
    https://github.com/zhenxun-zhuang/AdamW-Scale-free/blob/main/src/data_loader.py
"""
import torchvision
from torchvision import transforms


def get_cifar10(split, path): 
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])

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
    else:
        
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            normalize,
                                            ])
        
        ds = torchvision.datasets.CIFAR10(root=path, train=False, 
                                          download=True, 
                                          transform=transform_val
                                          )
        
    return ds


def get_cifar100(split, path): 
    normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])


    if split == 'train':
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
            
        ds = torchvision.datasets.CIFAR100(root=path, train=True, 
                                          download=True,
                                          transform=transform_train
                                          )
    else:
        
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            normalize,
                                            ])
        
        ds = torchvision.datasets.CIFAR100(root=path, train=False, 
                                          download=True, 
                                          transform=transform_val
                                          )
        
    return ds