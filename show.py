from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

from stepback.record import Record


exp_id = 'test1' # file name of config

R = Record(exp_id)
df = R.base_df # mean over runs
id_df = R.id_df


R.plot_metric(s='val_score', log_scale=False)
