from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy

from stepback.record import Record, id_to_dict


exp_id = 'test1' # file name of config

R = Record(exp_id)
df = R._build_base_df(agg='mean') # mean over runs

#%% aesthetics

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

score_names = {'train_loss': 'Training loss', 'val_loss': 'Validation loss', 'train_score': 'Training score', 'val_score': 'Validation score',
               'model_norm': r'$\|x^k\|$'}
col_dict = {'sgd': '#7fb285' , 'adam': '#f34213', 'momo': '#023047', 'sgd-m': '#de9151'}
#7fb285
markevery_dict = {'momo': 5, 'sgd': 15, 'adam': 10, 'sgd-m': 8}


#%% plotting

s = 'val_score' # what to plot
log_scale = False # 


fig, ax = plt.subplots()

for m in df.index.unique():
    x = df.loc[m,'epoch']
    y = df.loc[m,s]
    conf = id_to_dict(m) 
    
    label = conf['name'] + ', ' + r'$\alpha_0=$' + conf['lr']
    for k,v in conf.items():
        if k in ['name', 'lr']:
            continue
        label += ', ' + k + '=' + str(v)
    
    ax.plot(x, y, c=col_dict[conf['name']], marker='o', markersize=5, markevery=(markevery_dict[conf['name']], 20), label=label)

ax.set_xlabel('Epoch')
ax.set_ylabel(score_names[s])
ax.grid(which='both', lw=0.2, ls='--', zorder=-10)

if log_scale:
    ax.yscale('log')    
    
ax.legend(fontsize=8)
