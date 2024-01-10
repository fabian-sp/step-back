## Remarks on config management

1) The simple option: Create a dict-type config (e.g. like [test.json](test.json)). The file name (in this example we use ``my_exp.json``) will serve as an identifier ``exp_id`` in the next steps. You can then run all entries of the config with one job.

2) The more complicated, but versatile option (e.g. when one single run is expensive): You can split a dict-type config into subsets (which are then stored as temporary list-type configs). This allows to only create configs which have not been run before. 

*Case a)* Assume we want to rerun everything. Choose a `job_name` which will serve as folder name for temporary config files. Specify `splits` as the number of splits you wish (if not specified, it splits into lists of length one).

```python
from stepback.utils import split_config
split_config(exp_id='my_exp', job_name=job_name, config_dir='configs/', splits=None, only_new=False)
```


*Case b)* Assume you have already ran some settings and only want to run new settings. The function will determine whether a specific setting has been run, by looking into the output from a ``output_dir`` which belong to ``exp_id``. **This is an experimental feature and should be handled with caution!**. You can run

```python
from stepback.utils import split_config
split_config(exp_id='my_exp', job_name=job_name, config_dir='configs/', splits=None, only_new=True, output_dir='output/')
```


In both cases, this will create temporary list-type config files, stored in `configs/job_name/`, which can then be launched separately.
The temproary config files will follow the name pattern

```
my_exp-001.json
my_exp-002.json
my_exp-003.json
...
```
