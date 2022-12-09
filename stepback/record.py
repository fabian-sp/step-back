from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import itertools

from .log import Container

#%%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

score_names = {'train_loss': 'Training loss', 
               'val_loss': 'Validation loss', 
               'train_score': 'Training score', 
               'val_score': 'Validation score',
               'model_norm': r'$\|x^k\|$',
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
class Record:
    def __init__(self, exp_id: str, output_dir='output/', as_json=True):
        self.exp_id = exp_id
        self.C = Container(name=exp_id, output_dir=output_dir, as_json=as_json)
        
        print(f"Loading data from {output_dir+exp_id}")
        self.C.load() # load data
        
        self.raw_df = self._build_raw_df()
        self.id_df = self._build_id_df()
        self.base_df = self._build_base_df(agg='mean')
        return
    
    def _build_raw_df(self):
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
        return df
    
    def _build_id_df(self):
        # create columns with single parts of id
        id_cols = list()
        all_ids = self.raw_df.id.unique()
        for i in all_ids:
            d = id_to_dict(i)
            id_cols.append(d)
        
        id_df = pd.DataFrame(id_cols, index=all_ids)
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
            
            # create columns with single parts of id
            id_cols = list()
            for i in df.index:
                d = id_to_dict(i)
                id_cols.append(d)
            
            id_col_df = pd.DataFrame(id_cols, index=df.index)
            df = pd.concat([df,id_col_df], axis=1)
        
        elif agg == 'first':
            df = raw_df.groupby(['id', 'epoch'], sort=False).first()
            assert len(df.run_id.unique()) == 1
            df = df.drop('run_id',axis=1)
            df = df.reset_index(level=-1) # moves epoch out of index
        
        df = df.sort_values(['id', 'epoch'])
        return df
    
    #============ PLOTTING =================================
    #=======================================================
    def plot_metric(self, s, df=None, log_scale=False, ylim=None, ax=None):
        
        if df is None:
            df = self.base_df.copy()
        
        # has to be set freshly every time
        for m in aes.keys():
            aes[m]['marker_cycle'] = itertools.cycle(('o', 'p', 's', '>', 'v', 'D'))  
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
            
        for m in df.index.unique():
            x = df.loc[m,'epoch']
            y = df.loc[m,s]
            conf = id_to_dict(m) 
            
            # construct label
            label = conf['name'] + ', ' + r'$\alpha_0=$' + conf['lr']
            for k,v in conf.items():
                if k in ['name', 'lr']:
                    continue
                label += ', ' + k + '=' + str(v)
            
            # plot
            ax.plot(x, y, 
                    c=aes.get(conf['name'], aes['default']).get('color'), 
                    marker=next(aes.get(conf['name'], aes['default']).get('marker_cycle')), 
                    markersize=6, 
                    markevery=(aes.get(conf['name'], aes['default']).get('markevery'), 20), 
                    label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(score_names[s])
        ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
        
        if log_scale:
            ax.yscale('log')    
        if ylim:
            ax.set_ylim(ylim)
            
        ax.legend(fontsize=8, loc='lower left')

        return 

    
def id_to_dict(id):
    """utility for creating a dictionary from the identifier"""
    tmp = id.split(':')
    d = dict([j.split('=') for j in tmp])
    return d