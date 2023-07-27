import torch
from torch import nn

import math

"""
VGG architectures for CIFAR-10 and CIFAR-100 and Imagenet32

Adapted from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py

For CIFAR-100 and Imagenet32, we simply use the same architecture as for CIFAR-10 but with the last layer adapted.
"""

class VGG_CIFAR(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def _get_vgg(name, batch_norm=False, num_classes=10):
    if name == 'vgg11':
        m = VGG_CIFAR(make_layers(cfg['A'], batch_norm=batch_norm), num_classes=num_classes)
    elif name == 'vgg13':
        m = VGG_CIFAR(make_layers(cfg['B'], batch_norm=batch_norm), num_classes=num_classes)
    elif name == 'vgg16':
        m = VGG_CIFAR(make_layers(cfg['D'], batch_norm=batch_norm), num_classes=num_classes)
    elif name == 'vgg19':
        m = VGG_CIFAR(make_layers(cfg['E'], batch_norm=batch_norm), num_classes=num_classes)
    
    return m

def get_cifar_vgg(name, batch_norm=False, num_classes=10):
    assert num_classes in [10,100]
    m = _get_vgg(name, batch_norm, num_classes)
    return m

# def get_imagenet32_vgg(name, batch_norm=False):
#     m = _get_vgg(name, batch_norm, num_classes=1000)
#     return m
