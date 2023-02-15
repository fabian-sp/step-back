from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

from stepback.record import Record, score_names, id_to_dict, create_label


exp_id = 'cifar10_vgg16' # file name of config

R = Record(exp_id)
raw_df = R.raw_df 
base_df = R.base_df # mean over runs
id_df = R.id_df # dataframe with the optimizer setups that were run


fig = R.plot_metric(s='val_score', log_scale=False, legend=True)
save = False

## if you only want to plot good settings:
#ixx =  base_df[base_df['val_score'] >= 0.5].id.unique()
#df1 = base_df.loc[base_df.id.isin(ixx),:]
#fig = R.plot_metric(df=df1, s='val_score', log_scale=False, legend=False)
#fig.savefig('output/plots/' + exp_id + f'/all_{s}.pdf')

#%%
%matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%% stability plots

s = 'val_score' # the score that should be the y-axis
xaxis = 'lr' # the parameter that should be the x-axis
sigma = 1 # number of standard deviations to show (in one direction)
cutoff = None

#############
grouped = base_df.groupby(['name', xaxis])
max_epoch = grouped['epoch'].max()
assert len(max_epoch.unique()) == 1, "It seems that different setups ran for different number of epochs."

if cutoff is None:
    cutoff_epoch = (max_epoch[0], max_epoch[0])

# filter epochs
sub_df = base_df[(base_df.epoch >= cutoff_epoch[0]) & (base_df.epoch <= cutoff_epoch[1])] 
# group by all id_cols 
df = sub_df.groupby(list(id_df.columns))[s, s+'_std'].mean() # use dropna=False if we would have nan values
# move xaxis out of grouping
df = df.reset_index(level=xaxis)
# make xaxis float
df[xaxis] = df[xaxis].astype('float')
# get method and learning rate with best score
# best_ind, best_x = df.index[df[s].argmax()], df[xaxis][df[s].argmax()]

R._reset_marker_cycle()

fig, ax = plt.subplots(figsize=(6,5))
# .unique(level=) might be useful at some point
for m in df.index.unique():
    this_df = df.loc[m,:]
    this_df = this_df.sort_values(xaxis) # sort!
    name = this_df.index.get_level_values('name')[0]
    
    x = this_df[xaxis]
    y = this_df[s]
    y2 = this_df[s+'_std']
    
    label = ", ".join([k+"="+v for k,v in zip(df.index.names,m) if v != 'none'])
    ax.plot(x,y, c=R.aes.get(name, R.aes['default'])['color'], label=label,
            marker=next(R.aes.get(name, R.aes['default']).get('marker_cycle')), 
            #markevery=(1,5),
            )
    if sigma > 0:
        ax.fill_between(x, y-sigma*y2, y+sigma*y2,
                        color=R.aes.get(name, R.aes['default'])['color'],
                        alpha=0.1, zorder=-10)
    
    # mark overall best
    #if m == best_ind:
    #    ax.scatter(best_x, df[s].max(), s=40, marker='o', c='k', zorder=100)
        
if xaxis == 'lr':
    ax.set_xlabel('learning rate')
else:
    ax.set_xlabel(xaxis)

#ax.set_ylim(0,1)    
ax.set_ylabel(score_names[s])
ax.set_xscale('log')
ax.grid(axis='y', lw=0.2, ls='--', zorder=-10)
fig.legend(fontsize=9, loc='upper right')

#fig.tight_layout()
fig.subplots_adjust(top=0.8,bottom=0.09,left=0.1,right=0.97)
#grouped.indices.keys()

if save:
    fig.savefig('output/plots/' + exp_id + f'/stability_{xaxis}_{s}.pdf')

#%% plots the adaptive step size

df = R._build_base_df(agg='first')
df = df[df['name'] == 'momo']

counter = 0
ncol, nrow = 3,2

fig, axs = plt.subplots(nrow, ncol, figsize=(6.6,4))

for _id in df.id.unique():
    ax = axs.ravel()[counter]
    this_df = df[df.id == _id]
    
    iter_per_epoch = len(this_df['step_size_list'].iloc[0])
    upsampled = np.linspace(this_df.epoch.values[0], this_df.epoch.values[-1],\
                            len(this_df)*iter_per_epoch)
    
    # TODO: read the defaults from package
    if id_to_dict(_id)['beta']== 'none':
        _beta = 0.9
    else:
        _beta = float(id_to_dict(_id)['beta'])
     
    _bias_correction = id_to_dict(_id).get('bias_correction', 'none')
    if _bias_correction == 'none':
        _bias_correction = False
    elif _bias_correction == 'True':
        _bias_correction = True
    else:
        _bias_correction = False

    rho = 1 - _beta**(np.arange(len(this_df)*iter_per_epoch)+1)

    all_s = []
    all_s_median = []
    for j in this_df.index:
        all_s_median.append(np.median(this_df.loc[j,'step_size_list']))
        all_s += this_df.loc[j,'step_size_list'] 
    
    # plot adaptive term
    ax.scatter(upsampled, all_s, c='#023047', s=5, alpha=0.35)
    ax.plot(this_df.epoch, all_s_median, c='gainsboro', marker='o', markevery=(5,7),\
            markerfacecolor='#023047', markeredgecolor='gainsboro', lw=2.5, label=r"$\zeta_k$")
    

    # plot LR
    if _bias_correction:
        y = np.repeat(this_df.learning_rate, iter_per_epoch) / rho
        ax.plot(upsampled, y, c='silver', lw=2.5, label=r"$\alpha_k/\rho_k$")
    else:
        ax.plot(this_df.epoch, this_df.learning_rate, c='silver', lw=2.5, label=r"$\alpha_k$")
    #ax.plot(this_df.epoch, this_df.lr, c='silver', lw=2.5, label=r"$\alpha_k$") # OLD
    
    ax.set_xlim(0, )
    ax.set_ylim(1e-5, 1e3)
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
    
    ax.legend(loc='upper right', fontsize=10)
    
    
    ax.set_title(create_label(_id, subset=['lr','beta']), fontsize=8)
    
    counter += 1

if save:
    fig.savefig('output/plots/' + exp_id + f'/step_sizes.png', dpi=500)

