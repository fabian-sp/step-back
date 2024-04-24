"""
Script for generating plots.
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse

from stepback.record import Record
from stepback.utils import get_output_filenames
from stepback.plotting import plot_stability, plot_step_sizes

################# Main setup ###############################
parser = argparse.ArgumentParser(description='Generate step-back plots.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test', help="The id of the config (its file name).")
args = parser.parse_args()


try:
    exp_id = args.id
    save = True
except:
    exp_id = 'cifar100_resnet110'
    save = False

output_names = get_output_filenames(exp_id)
############################################################

#%%
#%matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%
R = Record(output_names)

R.filter(drop={'name': ['momo-adam-star', 'momo-star']})
R.filter(drop={'name': ['adabelief', 'adabound', 'lion', 'prox-sps']}) 
R.filter(keep={'lr_schedule': 'constant'})                          # only show constant learning rate results


base_df = R.base_df                                 # base dataframe for all plots
id_df = R.id_df                                     # dataframe with the optimizer setups that were run

# _ = R.plot_metric(s='val_score', log_scale=False, legend=True)

#%% plot training curves for a subset of runs:

# takes 3 best runs per methods
best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['val_score'].nlargest(3)
#best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['train_loss'].nsmallest(3)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx),:]

y0 = 0.3 if exp_id=='cifar100_resnet110' else 0.4 if exp_id=='cifar10_vit' else 0.6

fig, ax = R.plot_metric(df=df1, s='val_score', ylim=(y0, 1.05*df1.val_score.max()), log_scale=False, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.16,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_val_score.pdf')

fig, ax = R.plot_metric(df=df1, s='train_loss', log_scale=True, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.17,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_train_loss.pdf')


#%% stability plots

FIGSIZE = (4.8,3.2)

fig, axs = plot_stability(R, score='val_score', xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=FIGSIZE, save=save)
fig, axs = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=FIGSIZE, save=save)
fig, axs = plot_stability(R, score=['train_loss', 'val_score'], xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=(4.8,6.4), save=save)


#%% plots the adaptive step size
### THIS PLOT IS ONLY RELEVANT FOR METHODS WITH ADAPTIVE STEP SIZE
###################################

if exp_id == 'cifar10_resnet20':
    _ = plot_step_sizes(R, method='momo', grid=(3,3), start=None, stop=None, save=save)
    _ = plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=None, save=save)
elif exp_id == 'cifar10_vgg16':
    _ = plot_step_sizes(R, method='momo', grid=(3,3), start=2, stop=11, save=save)
    _ = plot_step_sizes(R, method='momo-adam', grid=(3,3), start=2, stop=11, save=save)
elif exp_id == 'mnist_mlp':
    _ = plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=None, save=save)
    _ = plot_step_sizes(R, method='momo-adam', grid=(3,2), start=None, stop=None, save=save)
elif exp_id == 'cifar100_resnet110':
    _ = plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=7, save=save)
    _ = plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=7, save=save)
elif exp_id == 'cifar10_vit':
    _ = plot_step_sizes(R, method='momo', grid=(2,2), start=1, stop=5, save=save)
    _ = plot_step_sizes(R, method='momo-adam', grid=(2,2), start=None, stop=None, save=save)


# %%
