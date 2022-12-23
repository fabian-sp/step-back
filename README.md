# step-back

Package for running and benchmarking Pytorch optimizers.


## Getting started

Install via 

    python setup.py

or in order to install in developer mode via

    python setup.py clean --all develop clean --all

## How to use

Any experiment needs a config file, see e.g. `configs/test1.json`.

* In the config you can specify at each key a list or a single entry. For every list entry, a cartesian product will be run.
* The same is true for the hypeprparameters of each entry in the `opt` key of the config file.
* Multiple runs can be done using the key `n_runs`. In each run the seed for shuffling the `DataLoader` changes.
* The name of the config file serves as experiment ID, used later for running and storing the output. 

You can run an experiment with `run.py` or with `run.ipynb`.

The output is stored in `output` if no other directory is specified. 