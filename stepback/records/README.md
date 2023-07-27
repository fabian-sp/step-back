# Records

For some experiments, all results are stored in csv format here.

The idea is: if you want to compare a new optimizer, you only run the new method, and no need to rerun the standard benchmarks like SGD or Adam. The model and dataset code is also available in `step-back`.



Glossary:

* `grad_norm` is the norm of the stochastic (mini batch) gradient at end of epoch.
* `model_norm` is the Euclidean norm of all model parameters.
* `_std` indicates standard deviation of a metric over the repeated runs (ie different `DataLoader` seeds).

