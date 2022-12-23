from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

from stepback.record import Record, score_names, id_to_dict, create_label


exp_id = 'cifar_vgg_2' # file name of config

R = Record(exp_id)
raw_df = R.raw_df 
base_df = R.base_df # mean over runs
id_df = R.id_df # dataframe with the optimizer setups that were run


R.plot_metric(s='val_score', log_scale=False)

#%% stability plots

s = 'val_score' # the score that should be the y-axis
xaxis = '_lr' # the parameter that should be the x-axis
cutoff = None

#############
grouped = base_df.groupby(['_name', xaxis])
max_epoch = grouped['epoch'].max()
assert len(max_epoch.unique()) == 1, "It seems that different setups ran for different number of epochs."

if cutoff is None:
    cutoff_epoch = (max_epoch[0], max_epoch[0])

# filter epochs
sub_df = base_df[(base_df.epoch >= cutoff_epoch[0]) & (base_df.epoch <= cutoff_epoch[1])] 
# group by all id_cols 
df = sub_df.groupby(list(id_df.columns))[s].mean()
# move xaxis out of grouping
df = df.reset_index(level=xaxis)
# make xaxis  float
df[xaxis] = df._lr.astype('float')
# get method and learning rate with best score
best_ind, best_x = df.index[df[s].argmax()], df[xaxis][df[s].argmax()]


fig, ax = plt.subplots(figsize=(6,4))
for m in df.index.unique():
    this_df = df.loc[m,:]
    this_df = this_df.sort_values(xaxis) # sort!
    name = this_df.index.get_level_values('_name')[0]
    
    x = this_df[xaxis]
    y = this_df[s]

    ax.plot(x,y, c=R.aes.get(name, R.aes['default'])['color'], label=str(m),
            #markevery=(1,5), marker='o'
            )
    
    # mark overall best
    if m == best_ind:
        ax.scatter(best_x, df[s].max(), s=40, marker='o', c='k', zorder=100)
        
if xaxis == '_lr':
    ax.set_xlabel('learning rate')
else:
    ax.set_xlabel(xaxis)
    
ax.set_ylabel(score_names[s])
ax.set_xscale('log')
ax.grid(axis='y', lw=0.2, ls='--', zorder=-10)
fig.legend(fontsize=9, loc='lower left')

fig.tight_layout()
#grouped.indices.keys()

#%% plots the adaptive step size

df = R._build_base_df(agg='first')
df = df[df['_name'] == 'momo']

counter = 0
ncol = 3
nrow = 2

fig, axs = plt.subplots(nrow, ncol, figsize=(10,6))

for _id in df.id.unique():
    ax = axs.ravel()[counter]
    this_df = df[df.id == _id]
    
    iter_per_epoch = len(this_df['step_size_list'].iloc[0])
    upsampled = np.linspace(this_df.epoch.values[0], this_df.epoch.values[-1],\
                            len(this_df)*iter_per_epoch)
    
    all_s = []
    all_s_median = []
    for j in this_df.index:
        all_s_median.append(np.median(this_df.loc[j,'step_size_list']))
        all_s += this_df.loc[j,'step_size_list'] 
    
    # plot adaptive term
    ax.scatter(upsampled, all_s, c='#023047', s=5, alpha=0.35)
    ax.plot(this_df.epoch, all_s_median, c='gainsboro', marker='o', markevery=(5,7),\
            markerfacecolor='#023047', markeredgecolor='gainsboro', lw=2.5, label=r"$\tau_k^+$")
    
    # plot LR
    ax.plot(this_df.epoch, this_df.learning_rate, c='silver', lw=2.5, label=r"$\alpha_k$")
    #ax.plot(this_df.epoch, this_df.lr, c='silver', lw=2.5, label=r"$\alpha_k$") # OLD
    
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
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, )
    ax.set_ylim(1e-3, 1e3)
    ax.set_yscale('log')
    
    ax.set_title(create_label(_id, subset=['beta']), fontsize=8)
    
    counter += 1