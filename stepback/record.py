from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import itertools

from .log import Container

#%%

score_names = {'train_loss': 'Training loss', 
               'val_loss': 'Validation loss', 
               'train_score': 'Training score', 
               'val_score': 'Validation score',
               'model_norm': r'$\|x^k\|$',
               'grad_norm': r'$\|g_k\|$'
               }


aes = {'sgd': {'color': '#7fb285', 'markevery': 15},
        'sgd-m': {'color': '#de9151', 'markevery': 8},
        'adam': {'color': '#f34213', 'markevery': 10}, 
        'adamw': {'color': '#f34213', 'markevery': 10},
        'momo': {'color': '#023047', 'markevery': 5},
        'default': {'color': 'grey', 'markevery': 3},
        }
#7fb285


#%%

_USE_UNDERSCORE = False # whether to add underscore to column names in id_df?

class Record:
    def __init__(self, exp_id: str, output_dir='output/', as_json=True):
        self.exp_id = exp_id
        self.aes = copy.deepcopy(aes)
        self.C = Container(name=exp_id, output_dir=output_dir, as_json=as_json)
        
        print(f"Loading data from {output_dir+exp_id}")
        self.C.load() # load data
        
        self.raw_df = self._build_raw_df()
        self.id_df = self._build_id_df()
        self.base_df = self._build_base_df(agg='mean')
        return
    
    def _build_raw_df(self):
        """ create DataFrame with the stored output. Creates an id column based on opt config. """
        df_list = list()
        for r in self.C.data:
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
            self.aes[m]['marker_cycle'] = itertools.cycle(('o', 'p', 's', '>', 'v', 'D'))  
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
                    c=aes.get(conf['name'], self.aes['default']).get('color'), 
                    marker=next(self.aes.get(conf['name'], self.aes['default']).get('marker_cycle')), 
                    markersize=6, 
                    markevery=(self.aes.get(conf['name'], self.aes['default']).get('markevery'), 20), 
                    label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(score_names[s])
        ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
        
        if log_scale:
            ax.set_yscale('log')    
        if ylim:
            ax.set_ylim(ylim)
            
        if legend:
            ax.legend(fontsize=8, loc='lower left')

        return 

    
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