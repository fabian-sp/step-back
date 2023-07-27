# Datasets

This page collects information how to retrieve the datasets.

* MNIST and CIFAR datasets are automatically downloaded by Pytorch. By default, they should be extracted to the `data` directory.
* Imagenet32 needs to be downloaded form the official Imagenet page (you need to create an account for this). The retrieved files should contain a folder `train` and a folder `val`. Extract them into a folder structure like below:

```
.
└── data/
    └── imagenet32/
        ├── train/
        │   ├── train_data_batch_1
        │   └── ...
        └── val/
        │   ├── val_data
```