from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
from ast import literal_eval as make_tuple
import copy

from .record import SCORE_NAMES, id_to_dict, create_label

###################################
### STABILITY
###################################


def plot_stability(R, 
                   score='val_score', 
                   xaxis='lr', 
                   sigma=1, 
                   cutoff=None,
                   ignore_columns=list(), 
                   legend=None, 
                   ylim=None, 
                   log_scale=True, 
                   figsize=(6,5), 
                   save=False
                   ):
    """
    Generates stability plot.

    Arguments:
        score: name of the score on y-axis, for example 'val_score' or 'train_loss'
        xaxis: parameter to group by, for example 'lr' for initial learning rate
        sigma: number of standard deviations to show (in one direction)
        cutoff: if not None, score is aggregated over [cutoff, max_epoch]
        ignore_columns: columns from id_df that are ignored for grouping. For example, useful when xaxis=lr but weight_decay is also different for each lr.
        legend: 'full', None, or a list of keys that are displayed, e.g. ['lr', 'weight_decay']
        ylim: Set ylim values. By default will be set to [0,1] for <_score> metrics
        log_scale: Use log-scale for <_loss> metrics. Not used for <_score> metrics

    """
    # plot only one score
    score = copy.deepcopy(score)
    if isinstance(score, str):
        score = [score]
    

    fig, axs = plt.subplots(len(score),1,figsize=figsize)

    for j, s in enumerate(score):
        df = R.build_sweep_df(score=s,
                              xaxis=xaxis,
                              ignore_columns=ignore_columns,
                              cutoff=cutoff
        )

        ax = axs.ravel()[j] if len(score) > 1 else axs 
        # .unique(level=) might be useful at some point

        # create ls cycle for each name
        all_name = list(df.index.unique(level='name'))
        ls_cycles = dict()
        for n in all_name:
            ls_cycles[n] = itertools.cycle(["-", '--', ':', '-.'])

        # main plotting
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
            
            ax.plot(x,y, 
                    c=R.aes.get(name, R.aes['default'])['color'], 
                    label=label,
                    marker="o",
                    ls=next(ls_cycles[name]),
                    zorder=R.aes.get(name, R.aes['default']).get('zorder')
                    )
            
            if sigma > 0:
                ax.fill_between(x, y-sigma*y2, y+sigma*y2,
                                color=R.aes.get(name, R.aes['default'])['color'],
                                alpha=0.1, 
                                zorder=-10)
            
            # mark overall best
            # if m == best_ind:
            #    ax.scatter(best_x, df[s].max(), s=40, marker='o', c='k', zorder=100)
                
        if xaxis == 'lr':
            ax.set_xlabel('Learning rate')
        else:
            ax.set_xlabel(xaxis)

        if ylim is not None:
            ax.set_ylim(ylim)
        elif s in ['train_score', 'val_score']:
            ax.set_ylim(0,1)

        if (s in ['train_loss', 'val_loss']) and log_scale:
            ax.set_yscale('log')

        ax.set_ylabel(SCORE_NAMES.get(s, s))
        ax.set_xscale('log')
        ax.grid(axis='y', lw=0.2, ls='--', zorder=-10)

        # xlabel only in last row
        if j+1 < len(score):
            ax.set_xlabel('')
            
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
    
    if save:
        fig.savefig('output/plots/' + R.exp_id_str + f"/stability_{xaxis}_" + f"{'_'.join(score)}.pdf")
    
    return fig, axs

def plot_single_step_sizes(this_df, aes, ax):
    method = this_df.name.iloc[0]
    _id = this_df.id.iloc[0]

    iter_per_epoch = len(this_df['step_size_list'].iloc[0])
    upsampled = np.linspace(this_df.epoch.values.min(), 
                            this_df.epoch.values.max(),
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
    ax.scatter(upsampled, 
               all_s, 
               c=aes[method]['color'], 
               s=5, 
               alpha=0.25)
    
    markevery = (5,7) if this_df.epoch.max() <= 100 else (5,20)
    
    # plot median markers, shift by 1/2 to get in middle of epoch
    ax.plot(this_df.epoch + 0.5, 
            all_s_median, 
            c='gainsboro', 
            marker='o', 
            markevery=markevery,
            markerfacecolor=aes[method]['color'], 
            markeredgecolor='gainsboro', 
            lw=2.5,
            label=r"$\zeta_k$")
    
    # plot LR
    y = np.repeat(this_df.learning_rate, iter_per_epoch)
    if rho is not None:
        y =  y/rho
        label = r"$\alpha_k/\rho_k$"
    else:
        label = r"$\alpha_k$"
    
    ax.plot(upsampled, y, c='silver', lw=2.5, label=label)
    
    return ax

###################################
### ADAPTIVE STEP SIZE
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
        
        ax = plot_single_step_sizes(this_df, R.aes, ax)

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
        fig.savefig('output/plots/'+ R.exp_id_str + f'/step_sizes_'+method+'.png', dpi=500)

    return fig, axs