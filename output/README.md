## Output folder

We store the results of all experiments here.

In the [plotting script](../show.py), for a given experiment ID `EXP_ID`, all output files in this folder are collected if their name is either

```
<EXP_ID>.json
<EXP_ID>-1.json, <EXP_ID>-2.json, ...
```

This has the following reason: it might be useful to split up config files even though they belong together. If we want to run parts of the same config in parallel, it should be safer to write to different output files. Hence, if desired, you can split your config into the same structure:

```
<EXP_ID>.json
<EXP_ID>-1.json, <EXP_ID>-2.json, ...
```