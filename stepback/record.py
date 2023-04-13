from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import itertools
from typing import Union
import warnings

from .log import Container

#%%

score_names = {'train_loss': 'Training loss', 
               'val_loss': 'Validation loss', 
               'train_score': 'Training score', 
               'val_score': 'Validation score',
               'model_norm': r'$\|x^k\|$',
               'grad_norm': r'$\|g_k\|$'
               }


aes = {'sgd': {'color': '#7fb285', 'markevery': 15, 'zorder': 7},
        'sgd-m': {'color': '#de9151', 'markevery': 8, 'zorder': 8},
        'adam': {'color': '#f34213', 'markevery': 10, 'zorder': 9}, 
        'adamw': {'color': '#f34213', 'markevery': 10, 'zorder': 9},
        'momo': {'color': '#023047', 'markevery': 5, 'zorder': 11},
        'momo-adam': {'color': '#3F88C5', 'markevery': 6, 'zorder': 10},
        'prox-sps': {'color': '#97BF88', 'markevery': 7, 'zorder': 6},
        'default': {'color': 'grey', 'markevery': 3, 'zorder': 1},
        }
# more colors:
#64B6AC
#F7CE5B
#4FB0C6
#3F88C5
#
#8be8cb
#7ea2aa

ALL_MARKER = ('o', 'H', 's', '>', 'v', '<' , '^', 'D', 'x')


#%%

_USE_UNDERSCORE = False # whether to add underscore to column names in id_df?

class Record:
    def __init__(self, 
                 exp_id: Union[str, list], 
                 output_dir='output/', 
                 as_json=True):
        
        self.exp_id = exp_id
        self.aes = copy.deepcopy(aes)

        # exp_id can be str or list (if we want to merge multiple output files)
        if isinstance(exp_id, str):
            exp_id = [exp_id]
        else:
            warnings.warn("Loading from multiple output files. Contents will be merged.")   
        
        # object to store everything
        self.data = list()

        for _e in exp_id:
            C = Container(name=_e, output_dir=output_dir, as_json=as_json)
            print(f"Loading data from {output_dir+_e}")
            C.load() # load data

            self.data += C.data # append

        self.raw_df = self._build_raw_df()
        self.id_df = self._build_id_df()
        self.base_df = self._build_base_df(agg='mean')
        return
    
    def filter(self, exclude=[]):

        base_df = self.base_df[~self.base_df.name.isin(exclude)]
        id_df = self.id_df[~self.id_df.name.isin(exclude)]

        return base_df, id_df

    def _build_raw_df(self):
        """ create DataFrame with the stored output. Creates an id column based on opt config. """
        df_list = list()
        for r in self.data:
            this_df = pd.DataFrame(r['history'])
            
            opt_dict = copy.deepcopy(r['config']['opt'])
            opt_dict = {'name': opt_dict.pop('name'), **opt_dict} # move name to beginning
            
            id = list()
            for k, v in opt_dict.items():       
                id.append(k+'='+ str(v)) 
                
            this_df['id'] = ':'.join(id) # identifier given by all opt specifications
            this_df['run_id'] = r['config']['run_id'] # differentiating multiple runs
            df_list.append(this_df)
            
        df = pd.concat(df_list)   
        df = df.reset_index(drop=True)
        df.insert(0, 'id', df.pop('id')) # move id column to front

        # raise error if duplicates
        if df.duplicated(subset=['id', 'epoch', 'run_id']).any():
            raise KeyError("There seem to be duplicates (by id, epoch, run_id). Please check the output data.")

        return df
    
    def _build_id_df(self):
        """ create a DataFrame where each id is split up into all hyperparameter settings """
        id_cols = list()
        all_ids = self.raw_df.id.unique()
        for i in all_ids:
            d = id_to_dict(i, add_underscore=_USE_UNDERSCORE)
            id_cols.append(d)
        
        id_df = pd.DataFrame(id_cols, index=all_ids)
        id_df.fillna('none', inplace=True)
        return id_df
        
    
    def _build_base_df(self, agg='mean'):
        raw_df = self.raw_df.copy()
        
        # compute mean for each id and(!) epoch
        if agg in ['mean', 'median']:
            df = raw_df.groupby(['id', 'epoch'], sort=False).mean().drop('run_id',axis=1)
            df2 = raw_df.groupby(['id', 'epoch'], sort=False).std().drop('run_id',axis=1)           
            df2.columns = [c+'_std' for c in df2.columns]
            
            df = pd.concat([df,df2], axis=1) 
            df = df.reset_index(level=-1) # moves epoch out of index
            
        elif agg == 'first':
            df = raw_df.groupby(['id', 'epoch'], sort=False).first()
            assert len(df.run_id.unique()) == 1
            df = df.drop('run_id', axis=1)
            df = df.reset_index(level=-1) # moves epoch out of index
        
        df = df.reset_index(drop=False) # set index as integers
        df = df.merge(self.id_df, how='left', left_on='id', right_index=True) # add columns from id_df
        
        return df
    
    #============ PLOTTING =================================
    #=======================================================
    def _reset_marker_cycle(self):
        for m in self.aes.keys():
            self.aes[m]['marker_cycle'] = itertools.cycle(ALL_MARKER)  
        return
    
    def plot_metric(self, s, df=None, log_scale=False, ylim=None, legend=True, ax=None):
        
        if df is None:
            df = self.base_df.copy()
        
        # has to be set freshly every time
        self._reset_marker_cycle()
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if legend:
            alpha = 1
            markersize = 6
        else:
            alpha = .65
            markersize = 4
            
        for m in df.id.unique():
            this_df = df[df.id==m]
            x = this_df.loc[:,'epoch']
            y = this_df.loc[:,s]
            conf = id_to_dict(m, add_underscore=False) 
            
            # construct label
            label = conf['name'] + ', ' + r'$\alpha_0=$' + conf['lr']
            for k,v in conf.items():
                if k in ['name', 'lr']:
                    continue
                label += ', ' + k + '=' + str(v)
            
            # plot
            ax.plot(x, y, 
                    c=self.aes.get(conf['name'], self.aes['default']).get('color'), 
                    marker=next(self.aes.get(conf['name'], self.aes['default']).get('marker_cycle')) if legend else 'o', 
                    markersize=markersize, 
                    markevery=(self.aes.get(conf['name'], self.aes['default']).get('markevery'), 20), 
                    alpha = alpha,
                    label=label,
                    zorder=self.aes.get(conf['name'], self.aes['default']).get('zorder'))
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(score_names[s])
        ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
        
        if log_scale:
            ax.set_yscale('log')    
        if ylim:
            ax.set_ylim(ylim)
        
        # full legend or only solver names
        if legend:
            ax.legend(fontsize=8, loc='lower left').set_zorder(100)
        else:
            for n in df.name.unique():
                handles = [Line2D([0], [0], color=aes.get(n, self.aes['default']).get('color'), lw=4) for n in df.name.unique()]
                names = list(df.name.unique())
                ax.legend(handles, names, loc='lower left').set_zorder(100)
        return fig

    
def id_to_dict(id, add_underscore=False):
    """utility for creating a dictionary from the identifier"""
    tmp = id.split(':')
    if add_underscore:
        tmp = ['_'+t for t in tmp] # let each column start with _ to indicate that it is added afterwards
    d = dict([j.split('=') for j in tmp])
    return d


def create_label(id, subset=None):
    d = id_to_dict(id)
    if subset is None:
        s = [key_to_math(k) +'='+ v for k,v in d.items()]
    else:
        s = [key_to_math(k) +'='+ v for k,v in d.items() if k in subset]
        
    return ', '.join(s)

def key_to_math(k):
    """translates column names from id_dict to math symbol"""
    if k == 'lr':
        k2 = r'$\alpha_0$'
    elif k == 'beta':
        k2 = r'$\beta$'
        #v2 = None if v == 'none' else float(v)
    if k == 'weight_decay':
        k2 = r'$\lambda$'
    return k2