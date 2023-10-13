# Models Overview

In the config, the key ``'model'`` specifies which model should be trained. ``step-back`` currently has the following models implemented.

In the below list, we first state the value for the key ``'model'`` and then give a short description. See [the main model script](main.py) for more details.

* ``resnet20``, ``resnet32`` etc.: ResNet model for CIFAR. There is an option to deactivate batch norm by setting ``"model_kwargs": {"batch_norm": false}``. Compatible with the following datasets: ``'cifar10','cifar100'``.

* ``vgg13``, ``vgg16`` etc.: VGG model for CIFAR. There is an option to deactivate batch norm by setting ``"model_kwargs": {"batch_norm": false}``. Compatible with the following datasets: ``'cifar10','cifar100'``.

* ``resnet18-kuangliu``, etc.: ResNet model for Imagenet32. The size of convolutional layers is slightly different to the ResNet models above. Compatible with the following datasets: ``'imagenet32'``.

* ``resnet18-pytorch``, ``resnet50-pytorch``: Official Pytorch ResNet model for Imagenet-1K. Compatible with the following datasets: ``'imagenet'``.

* ``vit``, ``swint``: Vision transformer and Swin Transformer. Compatible with possibly all image datasets, tested on ``cifar10`` so far.

* ``mlp``: MLP with ReLU activations. Make sure to specify the dimension of the last layer (before loss function) with ``"model_kwargs": {"output_size": }`` (e.g. use number of classes for classification). More options can be specified via 
``"model_kwargs": {"hidden_sizes": [128,64], "bias": true}``.
Compatible with any dataset where one input sample is a 1-D array.

* ``matrix_fac``: Matrix factorization with two layers. Input and output dimension are automatically inferred from the dataset. Compatible with following datasets: ``'synthetic_matrix_fac'``.

* ``matrix_completion``: Matrix completion with two layers. Matrix dimension (and rank) have to be specified in ``"model_kwargs": {"dim": (dim1,dim2), "rank": }``. Compatible with following datasets: ``'sensor'``.

* ``linear``: Linear model. By default, assumes that output is a scalar. Optionally, multidimensional output and bias can be specified with ``"model_kwargs": {"output_size": , "bias": true}``. Compatible with any dataset where one input sample is a 1-D array.

