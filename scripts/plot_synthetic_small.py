from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

os.chdir('..')

from stepback.record import Record, SCORE_NAMES
from stepback.utils import get_output_filenames
from stepback.plotting import plot_stability, plot_step_sizes

exp_id = 'synthetic_small'
save = False

output_names = get_output_filenames(exp_id)
############################################################

#%%
%matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%
R = Record(output_names)


base_df = R.base_df                                 # base dataframe for all plots
id_df = R.id_df                                     # dataframe with the optimizer setups that were run

#fig = R.plot_metric(s='val_score', log_scale=False, legend=True)

#%% plot training curves for a subset of runs:

# takes 3 best runs per methods
best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['train_loss'].nsmallest(4)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx),:]

fig = R.plot_metric(df=df1, s='fstar', log_scale=False, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.16,right=0.975)

fig = R.plot_metric(df=df1, s='train_loss', log_scale=True, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.16,right=0.975)


#%%

FIGSIZE = (4.8,3.2)

fig = plot_stability(R, score='val_loss', xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=FIGSIZE, save=save)
fig = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=FIGSIZE, save=save)

# %%

plot_step_sizes(R, method='momo-star', grid=(3,2), start=None, stop=6, save=save)
plot_step_sizes(R, method='momo-adam-star', grid=(3,2), start=1, stop=7, save=save)