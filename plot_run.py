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


parser = argparse.ArgumentParser(description='Plot runs of stepback from the terminal.')
parser.add_argument('-ids','--ids', nargs='+', default='test1', help="The ids of the configs (its file name) you want to plot.", required=True)
parser.add_argument('-name', '--name', nargs='?', type=str, default='?', help="The name of the plots.")
parser.add_argument('-cdir', '--config_dir', nargs='?', type=str, default='configs/', help="The config directory.")
parser.add_argument('-odir', '--output_dir', nargs='?', type=str, default='output/', help="The output directory.")
''' 
For plotting the validation and training accuracy given a set of experiment ids

Example for running on terminal with multiply experiments:

    python3 plot_run.py --ids cifar10_vgg16 cifar10_vgg16-2 cifar10_vgg16-3 --name cifar10_vgg16


Example for running on terminal with one experiment:

    python3 plot_run.py --ids test1 --name test1
'''


# for running from IPython
CONFIG_DIR = 'configs/'
OUTPUT_DIR = 'output/'

def plot_run(exp_ids, name, output):
    R = Record(exp_ids)
    # raw_df = R.raw_df 
    # base_df = R.base_df # mean over runs
    # id_df = R.id_df # dataframe with the optimizer setups that were run
    # import pdb; pdb.set_trace()
    directory = output+'/plots/' + name
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)

    columns = ['val_score', 'train_score', 'train_loss']
    for col in columns:
        fig = R.plot_metric(s=col, log_scale=False, legend=False)
        fig.savefig(directory + '/' + col + '.pdf')
        fig.clear()

    return 

if __name__ == '__main__':
    args = parser.parse_args()

    CONFIG_DIR = args.config_dir
    OUTPUT_DIR = args.output_dir
    EXP_IDS = args.ids
    NAME = args.name

    if NAME =='?':
        NAME = EXP_IDS[0]
    plot_run(EXP_IDS,NAME,OUTPUT_DIR)
    

