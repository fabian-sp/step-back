"""
Script for generating plots.
"""
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import argparse
from ast import literal_eval as make_tuple

from stepback.record import Record, score_names, id_to_dict, create_label
from stepback.utils import get_output_filenames

################# Main setup ###############################
parser = argparse.ArgumentParser(description='Generate step-back plots.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test1', help="The id of the config (its file name).")
args = parser.parse_args()
exp_id = args.id
#exp_id = 'cifar100_resnet110'

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
base_df = R.base_df                                 # base dataframe for all plots
id_df = R.id_df                                     # dataframe with the optimizer setups that were run
base_df, id_df = R.filter(exclude=['momo-adam-star', 'momo-star', 'momo-adam-max', 'momo-max'])     # filter out a method

#fig = R.plot_metric(s='val_score', log_scale=False, legend=True)

#%% plot training curves for a subset of runs:

#ixx =  base_df[base_df['val_score'] >= 0.5].id.unique()
#df1 = base_df.loc[base_df.id.isin(ixx),:]

# takes 3 best runs per methods
best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['val_score'].nlargest(3)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx),:]

fig = R.plot_metric(df=df1, s='val_score', ylim = (0.6, 1.05*df1.val_score.max()), log_scale=False, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.155,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_val_score.pdf')

#%% stability plots

def plot_stability(base_df, score='val_score', xaxis='lr', sigma=1, cutoff=None, full_legend=True, figsize=(6,5), save=False):
    """
    Generates stability plot.

    Arguments:
        score: name of the score on y-axis, for example 'val_score' or 'train_loss'
        xaxis: parameter to group by, for example 'lr' for initial learning rate
        sigma: number of standard deviations to show (in one direction)
        cutoff: if not None, score is aggregated over [cutoff, max_epoch]


    """
    grouped = base_df.groupby(['name', xaxis])
    max_epoch = grouped['epoch'].max()
    assert len(max_epoch.unique()) == 1, "It seems that different setups ran for different number of epochs."

    if cutoff is None:
        cutoff_epoch = (max_epoch[0], max_epoch[0])
    else:
        cutoff_epoch = (cutoff, max_epoch[0])

    # filter epochs
    sub_df = base_df[(base_df.epoch >= cutoff_epoch[0]) & (base_df.epoch <= cutoff_epoch[1])] 
    # group by all id_cols 
    df = sub_df.groupby(list(id_df.columns))[score, score+'_std'].mean() # use dropna=False if we would have nan values
    # move xaxis out of grouping
    df = df.reset_index(level=xaxis)
    # make xaxis float
    df[xaxis] = df[xaxis].astype('float')
    # get method and learning rate with best score
    # best_ind, best_x = df.index[df[s].argmax()], df[xaxis][df[s].argmax()]

    R._reset_marker_cycle()

    fig, ax = plt.subplots(figsize=figsize)
    # .unique(level=) might be useful at some point
    for m in df.index.unique():
        this_df = df.loc[m,:]
        this_df = this_df.sort_values(xaxis) # sort!
        name = this_df.index.get_level_values('name')[0]
        
        x = this_df[xaxis]
        y = this_df[score]
        y2 = this_df[score+'_std']
        
        if full_legend:
            label = name + ", " + ", ".join([k+"="+v for k,v in zip(df.index.names,m) if (v!='none') and (k!='name')])
        else:
            label = name
           
        ax.plot(x,y, c=R.aes.get(name, R.aes['default'])['color'], label=label,
                marker=next(R.aes.get(name, R.aes['default']).get('marker_cycle')), 
                zorder=R.aes.get(name, R.aes['default']).get('zorder')
                )
        
        if sigma > 0:
            ax.fill_between(x, y-sigma*y2, y+sigma*y2,
                            color=R.aes.get(name, R.aes['default'])['color'],
                            alpha=0.1, zorder=-10)
        
        # mark overall best
        #if m == best_ind:
        #    ax.scatter(best_x, df[s].max(), s=40, marker='o', c='k', zorder=100)
            
    if xaxis == 'lr':
        ax.set_xlabel('Learning rate')
    else:
        ax.set_xlabel(xaxis)

    if score == 'val_score':
        ax.set_ylim(0,1)    
    elif score == 'train_loss':
        ax.set_yscale('log')

    ax.set_ylabel(score_names[score])
    ax.set_xscale('log')
    ax.grid(axis='y', lw=0.2, ls='--', zorder=-10)
    
    if full_legend:
        # legend has all specific opt arguments
        fig.legend(fontsize=8, loc='upper right')
    else:
        # legend only has name
        fig.legend(fontsize=11, loc='upper right', 
                   ncol=min(len(ax.get_legend_handles_labels()[0]),4), columnspacing=0.6)

    fig.tight_layout()
    if full_legend:
        fig.subplots_adjust(top=0.75,bottom=0.125,left=0.14,right=0.97)
    else:
        fig.subplots_adjust(top=0.85,bottom=0.115,left=0.145,right=0.98)
    #grouped.indices.keys()

    if save:
        fig.savefig('output/plots/' + exp_id + f'/stability_{xaxis}_{score}.pdf')
    
    return fig

FIGSIZE = (4.8,4)

fig = plot_stability(base_df, score='val_score', xaxis='lr', sigma=1, full_legend=False, cutoff=None, figsize=FIGSIZE, save=save)
fig = plot_stability(base_df, score='train_loss', xaxis='lr', sigma=1, full_legend=False, cutoff=None, figsize=FIGSIZE, save=save)

#%% plots the adaptive step size
### THIS PLOT IS ONLY RELEVANT FOR METHODS WITH ADAPTIVE STEP SIZE
###################################

def plot_step_sizes(R, method='momo', ylim=(1e-5,1e3), xlim = None, grid=(3,3), figsize=None, start=None, stop=None, save=False):
    nrow, ncol = grid
    if figsize is None:
        figsize = (ncol*2,nrow*1.5)
    
    df = R._build_base_df(agg='first').copy()
    df = df[df['name'] == method]
    # make lr to float and sort
    df.lr = df.lr.astype(float)
    df = df.sort_values(['lr', 'epoch'], ascending=True)

    add_beta_to_title = False
    if 'beta' in df.columns:
        if len(df.beta.unique()) > 1:
            add_beta_to_title = True

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    counter = 0

    # filter
    ids_to_plot = df.id.unique()[start:stop]

    for _id in ids_to_plot:
        ax = axs.ravel()[counter]
        this_df = df[df.id == _id]
        
        iter_per_epoch = len(this_df['step_size_list'].iloc[0])
        upsampled = np.linspace(this_df.epoch.values[0], this_df.epoch.values[-1],\
                                len(this_df)*iter_per_epoch)
        
        if method in ['momo', 'momo-star']:
            # caution as id_df contains strings!
            _beta = 0.9 if id_to_dict(_id).get('beta', 'none') == 'none' else float(id_to_dict(_id)['beta'])
            _bias_correction = True if id_to_dict(_id).get('bias_correction') == 'True' else False
            rho = 1 - _bias_correction*_beta**(np.arange(len(this_df)*iter_per_epoch)+1)
        
        elif method in ['momo-adam', 'momo-adam-star']:
            _beta = 0.9 if id_to_dict(_id).get('betas', 'none') == 'none' else make_tuple(id_to_dict(_id)['betas'])[0]
            rho = 1 - _beta**(np.arange(len(this_df)*iter_per_epoch)+1)

        else:
            rho = None

        # compute median
        all_s = []
        all_s_median = []
        for j in this_df.index:
            all_s_median.append(np.median(this_df.loc[j,'step_size_list']))
            all_s += this_df.loc[j,'step_size_list'] 
        
        # plot adaptive term
        ax.scatter(upsampled, all_s, c=R.aes[method]['color'], s=5, alpha=0.25)
        ax.plot(this_df.epoch, all_s_median, 
                c='gainsboro', 
                marker='o', 
                markevery=(5,7),
                markerfacecolor=R.aes[method]['color'], 
                markeredgecolor='gainsboro', 
                lw=2.5,
                label=r"$\zeta_k$")
        
        # plot LR
        if rho is not None:
            y = np.repeat(this_df.learning_rate, iter_per_epoch) / rho
            ax.plot(upsampled, y, c='silver', lw=2.5, label=r"$\alpha_k/\rho_k$")
        else:
            ax.plot(this_df.epoch, this_df.learning_rate, c='silver', lw=2.5, label=r"$\alpha_k$")
        
        if xlim is None:
            ax.set_xlim(0, )
        else:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)
        ax.set_yscale('log')
        
        if counter%ncol == 0:
            ax.set_ylabel('Step size', fontsize=10)
            ax.tick_params(axis='y', which='major', labelsize=9)
            ax.tick_params(axis='y', which='minor', labelsize=6)    
        else:
            ax.set_yticks([])
            
        if counter >= ncol*(nrow-1):
            ax.set_xlabel('Epoch', fontsize=10)
            ax.tick_params(axis='x', which='both', labelsize=9)
        else:
            ax.set_xticks([])
        
        # plot legend only once
        if counter == 0:
            ax.legend(loc='upper right', fontsize=10)
        
        if method in ['momo','momo-star'] and add_beta_to_title:
            ax.set_title(create_label(_id, subset=['lr','beta']), fontsize=8)
        else:
            ax.set_title(create_label(_id, subset=['lr']), fontsize=8)

        counter += 1

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.tight_layout()

    if save:
        fig.savefig('output/plots/'+exp_id+f'/step_sizes_'+method+'.png', dpi=500)

    return fig

if exp_id == 'cifar10_resnet20':
    plot_step_sizes(R, method='momo', grid=(3,3), start=None, stop=None, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(2,3), start=1, stop=None, save=save)
elif exp_id == 'cifar10_vgg16':
    plot_step_sizes(R, method='momo', grid=(3,3), start=2, stop=11, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(3,3), start=2, stop=11, save=save)
elif exp_id == 'mnist_mlp':
    plot_step_sizes(R, method='momo', grid=(3,4), start=2, stop=None, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(3,2), start=None, stop=None, save=save)
elif exp_id == 'cifar100_resnet110':
    plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=7, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=7, save=save)