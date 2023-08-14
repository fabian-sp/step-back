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
base_df = R.base_df                                 # base dataframe for all plots
id_df = R.id_df                                     # dataframe with the optimizer setups that were run

base_df, id_df = R.filter(exclude=['momo-adam-star', 'momo-star',
                                    'adabelief', 'adabound', 'prox-sps'])     # filter out a method

#fig = R.plot_metric(s='val_score', log_scale=False, legend=True)

#%% plot training curves for a subset of runs:

# takes 3 best runs per methods
best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['val_score'].nlargest(3)
#best = base_df[base_df.epoch==base_df.epoch.max()].groupby('name')['train_loss'].nsmallest(3)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx),:]

y0 = 0.3 if exp_id=='cifar100_resnet110' else 0.4 if exp_id=='cifar10_vit' else 0.6

fig = R.plot_metric(df=df1, s='val_score', ylim=(y0, 1.05*df1.val_score.max()), log_scale=False, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.16,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_val_score.pdf')

fig = R.plot_metric(df=df1, s='train_loss', log_scale=True, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.17,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_train_loss.pdf')


#%% stability plots

def plot_stability(base_df, score='val_score', xaxis='lr', sigma=1, cutoff=None, legend=None, figsize=(6,5), save=False):
    """
    Generates stability plot.

    Arguments:
        score: name of the score on y-axis, for example 'val_score' or 'train_loss'
        xaxis: parameter to group by, for example 'lr' for initial learning rate
        sigma: number of standard deviations to show (in one direction)
        cutoff: if not None, score is aggregated over [cutoff, max_epoch]

    """
    # plot only one score
    if isinstance(score, str):
        score = [score]

    grouped = base_df.groupby(['name', xaxis])
    max_epoch = grouped['epoch'].max()
    assert len(max_epoch.unique()) == 1, "It seems that different setups ran for different number of epochs."

    if cutoff is None:
        cutoff_epoch = (max_epoch[0], max_epoch[0])
    else:
        cutoff_epoch = (cutoff, max_epoch[0])

    fig, axs = plt.subplots(len(score),1,figsize=figsize)

    for j, s in enumerate(score):
        # filter epochs
        sub_df = base_df[(base_df.epoch >= cutoff_epoch[0]) & (base_df.epoch <= cutoff_epoch[1])] 
        # group by all id_cols 
        df = sub_df.groupby(list(id_df.columns))[[s, s+'_std']].mean() # use dropna=False if we would have nan values
        # move xaxis out of grouping
        df = df.reset_index(level=xaxis)
        # make xaxis float
        df[xaxis] = df[xaxis].astype('float')
        
        # get method and learning rate with best score
        # best_ind, best_x = df.index[df[s].argmax()], df[xaxis][df[s].argmax()]

        R._reset_marker_cycle()
        ax = axs.ravel()[j] if len(score) > 1 else axs 
        # .unique(level=) might be useful at some point
        for m in df.index.unique():
            this_df = df.loc[m,:]
            this_df = this_df.sort_values(xaxis) # sort!
            name = this_df.index.get_level_values('name')[0]
            
            x = this_df[xaxis]
            y = this_df[s]
            y2 = this_df[s+'_std']
            
            if legend is None:
                label = name # default: only show method name
            elif legend == 'full':
                label = name + ", " + ", ".join([k+"="+v for k,v in zip(df.index.names,m) if (v!='none') and (k!='name')]) # show all keys
            else:
                label = name + ", " + ", ".join([k+"="+v for k,v in zip(df.index.names,m) if (v!='none') and (k!='name') and (k in legend)]) # show subset of keys
            
            label = label if j+1 == len(score) else None # avoid duplicate labels in legend
            
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

        if s in ['train_score', 'val_score']:
            ax.set_ylim(0,1)    
        elif s in ['train_loss', 'val_loss']:
            ax.set_yscale('log')

        ax.set_ylabel(score_names[s])
        ax.set_xscale('log')
        ax.grid(axis='y', lw=0.2, ls='--', zorder=-10)

        # xticks only in last row
        if j+1 < len(score):
            ax.set_xlabel('')
            #ax.set_xticks([])
    
    if legend is not None:
        # legend has all specific opt arguments
        fig.legend(fontsize=8, loc='upper right')
    else:
        # legend only has name
        fig.legend(fontsize=11, loc='upper right', 
                   ncol=min(len(ax.get_legend_handles_labels()[0]),4), columnspacing=0.6)

    fig.tight_layout()
    if legend is not None:
        fig.subplots_adjust(top=0.75,bottom=0.125,left=0.14,right=0.97)
    else:
        if len(score) > 1:
            fig.subplots_adjust(top=0.935,bottom=0.084,left=0.16,right=0.96)
        else:
            fig.subplots_adjust(top=0.855,bottom=0.155,left=0.145,right=0.98)
    
    # 0.805 for mnist
    #grouped.indices.keys()

    if save:
        fig.savefig('output/plots/' + exp_id + f"/stability_{xaxis}_{'_'.join(score)}.pdf")
    
    return fig

FIGSIZE = (4.8,3.2)

fig = plot_stability(base_df, score='val_score', xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=FIGSIZE, save=save)
fig = plot_stability(base_df, score='train_loss', xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=FIGSIZE, save=save)

fig = plot_stability(base_df, score=['train_loss', 'val_score'], xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=(4.8,6.4), save=save)


#%% plots the adaptive step size
### THIS PLOT IS ONLY RELEVANT FOR METHODS WITH ADAPTIVE STEP SIZE
###################################

def plot_single_step_sizes(this_df, ax):
    method = this_df.name.iloc[0]
    _id = this_df.id.iloc[0]

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
    markevery = (5,7) if this_df.epoch.max() <= 100 else (5,20)
    ax.plot(this_df.epoch, all_s_median, 
            c='gainsboro', 
            marker='o', 
            markevery=markevery,
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
    
    return ax

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
        
        ax = plot_single_step_sizes(this_df, ax)

        if xlim is None:
            ax.set_xlim(0, )
        else:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)
        ax.set_yscale('log')
        
        # zoomed in inset    
        # if counter==3:
        #     # inset axes....
        #     ax2 = ax.inset_axes([0.15, 0.02, 0.3, 0.47])
        #     ax2 = plot_single_step_sizes(this_df, ax2)
        #     ax2.set_xlim(0,0.5)
        #     ax2.set_ylim(ylim)
        #     ax2.set_yscale('log')
        #     ax2.set_yticks([]); ax2.set_xticks([])
        #     ax.indicate_inset_zoom(ax2, edgecolor="black", lw=2)


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
    plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=None, save=save)
elif exp_id == 'cifar10_vgg16':
    plot_step_sizes(R, method='momo', grid=(3,3), start=2, stop=11, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(3,3), start=2, stop=11, save=save)
elif exp_id == 'mnist_mlp':
    plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=None, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(3,2), start=None, stop=None, save=save)
elif exp_id == 'cifar100_resnet110':
    plot_step_sizes(R, method='momo', grid=(3,2), start=1, stop=7, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(3,2), start=1, stop=7, save=save)
elif exp_id == 'cifar10_vit':
    plot_step_sizes(R, method='momo', grid=(2,2), start=None, stop=None, save=save)
    plot_step_sizes(R, method='momo-adam', grid=(2,2), start=None, stop=None, save=save)


# %%
