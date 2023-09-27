"""
Creates a Latex table script with the best score for each method.
"""
import latextable
from texttable import Texttable
import numpy as np
import pandas as pd
import os

from stepback.record import Record
from stepback.utils import get_output_filenames

os.chdir('..')

ALL_EXP = ['cifar100_resnet110', 'cifar10_resnet20', 'cifar10_vgg16', 'mnist_mlp', 'cifar10_vit']
METHODS = ['momo', 'momo-adam', 'sgd-m', 'adam']
SCORE = 'val_score'

#exp_id = 'cifar100_resnet110'

BEST_IN_BOLD = True
ADD_STD = True
TO_PERCENT = True

method_map = {'momo': '\\texttt{MoMo}', 
              'momo-adam': '\\texttt{MoMo-Adam}', 
              'sgd-m': '\\texttt{SGD-M}',
              'adam': '\\texttt{Adam}'}

exp_map = {'cifar100_resnet110': '\\texttt{ResNet110} for \\texttt{CIFAR100}',
           'cifar10_resnet20': '\\texttt{ResNet20} for \\texttt{CIFAR10}',
           'cifar10_vgg16': '\\texttt{VGG16} for \\texttt{CIFAR10}',
           'mnist_mlp': '\\texttt{MLP} for \\texttt{MNIST}',
           'cifar10_vit': '\\texttt{ViT} for \\texttt{CIFAR10}'
          }

#%%
def get_best_score(exp_id):
    output_names = get_output_filenames(exp_id)
    
    R = Record(output_names)
    
    # filter
    R.filter(drop={'name': ['momo-adam-star', 'momo-star']})
    R.filter(drop={'name': ['adabelief', 'adabound', 'prox-sps']}) 
    R.filter(keep={'lr_schedule': 'constant'})                          # only show constant learning rate results

    df = R.base_df.copy()
    df = df[df.name.isin(METHODS)]

    if len(df.groupby('id')['epoch'].max().unique()) > 1:
        raise KeyError("Maximum epochs is not the same.")

    # filter to best per method
    best = df[df.epoch==df.epoch.max()].groupby('name')[SCORE].nlargest(1)
    T = df.loc[best.index.levels[1]] # table of best scores per method

    return T

def create_single_row(T):
    row = dict() 
    best_method = T[T.index == T[SCORE].idxmax()]['name'].item()

    for m in METHODS:
        
        if len(T[T.name==m]) == 0:
            row[m] = "NA" # no results exist
        else:
            val = (T[T.name==m][SCORE]).item()
            std = (T[T.name==m][SCORE + '_std']).item()

            # formatting
            if TO_PERCENT:
                val *= 100
                std *= 100
            
            val = np.round(val,2)
            std = np.round(std,2)


            suffix = rf'$~\pm {std}$' if ADD_STD else ''
            row[m] = str(val) +  suffix 

        if BEST_IN_BOLD and m == best_method:
            row[m] = '\\textbf{' + str(val) +  suffix + '}'

    return row

#%%

table = Texttable()
table.set_cols_align((len(METHODS)+1)*["c"])

header = [""] + [method_map[m] for m in METHODS]
table.add_row(header)

for exp_id in ALL_EXP:

    T = get_best_score(exp_id)
    this_row = create_single_row(T)
    table.add_row([exp_map[exp_id]] + list(this_row.values()))   

    #print(table.draw())

print(latextable.draw_latex(table, 
                            caption = "Validation score (with one standard deviation) for the best learning rate choice for each method. Best method in bold.", 
                            label = "table:best_scores"))

