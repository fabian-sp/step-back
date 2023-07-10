<img src="data/logo2.svg"
     alt="Step-back logo"
     width="500" />

Package for running and benchmarking Pytorch optimizers.


## Getting started

Install via 

    python setup.py

or in order to install in developer mode via

    python setup.py clean --all develop clean --all

## Results

For the experiments we ran, we provide the code that generated the results (i.e. the model, dataset preprocessing and training setup) as well as the actual scores at the end of each epoch. An overview and all links are given in the table below.

| ID  | Model  | Dataset  | Results  |  
|-----|--------|----------|----------|
| cifar100_resnet110  | ResNet110 [code](stepback/models/resnet.py)  | Cifar100 [code](stepback/datasets/cifar.py) | [record](stepback/records/cifar100_resnet110.csv)  |   
| cifar10_resnet20  | ResNet20 [code](stepback/models/resnet.py) | Cifar10 [code](stepback/datasets/cifar.py) | [record](stepback/records/cifar10_resnet20.csv)  |   
| cifar10_vgg16  | VGG16 [code](stepback/models/vgg.py) | Cifar10 [code](stepback/datasets/cifar.py) | [record](stepback/records/cifar10_vgg16.csv)  |   
| cifar10_vit  | Small ViT [code](stepback/models/vit/vit.py) | Cifar10 [code](stepback/datasets/cifar.py) | [record](stepback/records/cifar10_vit.csv)  |   
| mnist_mlp  |  MLP (3-layer, ReLU) [code](stepback/models/basic_models.py) | MNIST [code](stepback/datasets/mnist.py) | [record](stepback/records/mnist_mlp.csv)  |   


For each experiment, the exact config can also be found under `configs/` where the files are named according to the ID (and possibly an integer suffix).


## How to use

Any experiment needs a config file, see e.g. `configs/test1.json`.

* In the config you can specify at each key a list or a single entry. For every list entry, a cartesian product will be run.
* The same is true for the hypeprparameters of each entry in the `opt` key of the config file.
* Multiple runs can be done using the key `n_runs`. In each run the seed for shuffling the `DataLoader` changes.
* The name of the config file serves as experiment ID, used later for running and storing the output. 

You can run an experiment with `run.py` or with `run.ipynb`.

The output is stored in `output` if no other directory is specified.


## Output structure

Every single run stores output as a dictionary in the following way:

```
    {'config': configuration of the experiment
     'history': list of dictionary (one per epoch), see key names below
     'summary': useful information such as start time and end time
    } 
```

For the entries in `history`, the following keys are important:

* `learning_rate`: the learning rate value in that epoch. This is different to the `lr` key in `config['opt']` if learning rate schedule is used.
* `train_loss`: loss function value over training set
* `val_loss`: loss function value over validation set
* `train_score`: score function (eg accuracy) over training set
* `val_score`: score function (eg accuracy) over validation set