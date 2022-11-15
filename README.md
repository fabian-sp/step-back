# step-back

Package for runnign and benchmarking Pytorch optimizers.

An experiment needs a config file, see e.g. `configs/test1.json`.

* In the config you can specify at each key a list or a single entry. For every list entry, a cartesian product will be run.
* The same is true for the hypeprparameters of each entry in the `opt` key of the config file.
* Multiple runs can be done using the key `n_runs`. In eaach run the seed of the DataLoader changes.
* The name of the config file serves as experiment ID, used later for running and storing the output. 

You can run an experiment with `run.py`. (**TODO: run multiple experiments, in parallel?**)

The output is stored in `output`. Plotting utilities have to be added.

## Getting started

Install via 

    python setup.py

or in order to install in developer mode via

    python setup.py clean --all develop clean --all

