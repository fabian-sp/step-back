## Output folder

We store the results of all experiments here.

**Important:** The [``Record``](../stepback/record.py) object - which serves for plotting, analyzing results etc - will collect output from multiple files for a given experiment ID `EXP_ID`. Specifically, it loads the output from all files in this folder if the file name is in

```
<EXP_ID>.json
<EXP_ID>-1.json, <EXP_ID>-2.json, ...
```

We do this because it might be useful to split up output of different runs which actually *belong together* into different files.
You can however also easily merge multiple output files (or all files in a subdirectory) with the utilities in [`stepback.utils.py`](../stepback/utils.py).