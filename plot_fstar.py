"""
Main file for running experiments.
"""
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from stepback.record import Record, score_names, id_to_dict, create_label

output = 'output/'
exp_ids = 'mnist_mlp_small_rep' #  mnist_mlp_small_rep, rob_test2


name= exp_ids
R = Record(exp_ids)
df = R.base_df # mean over runs

directory = output+'/plots/' + name
isExist = os.path.exists(directory)
if not isExist:
    os.makedirs(directory)

columns = ['f_star', 'h']
colorset = {'f_star' : 'red' , 'h' : 'blue'}
colours = ['#de9151', '#f34213', '#1F86E0', '#97BF88', '#f34213','#023047', 'red', 'blue', 'green', '#f34213', '#1F86E0', '#97BF88', '#f34213','#023047', 'red', 'blue', 'green']
markers = ['o', 'H', 's', '>', 'v', '<' , '^', 'D', 'x','o', 'H', 's', '>', 'v', '<' , '^', 'D', 'x']
fig, ax = plt.subplots()
for col in columns:
    for m in df.id.unique():
        this_df = df[df.id==m]
        x = this_df.loc[:,'epoch']
        y = this_df.loc[:,col]
        # import pdb; pdb.set_trace()
        conf = id_to_dict(m, add_underscore=False) 
        label = col+'-'+conf['lr']
        # plot
        ax.plot(x, y, color=colours.pop(), marker = markers.pop(), 
                markersize=4,  
                alpha = 0.4,
                label=label)
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
fig.savefig(directory + '/' +'f_star-h' + '.pdf')





