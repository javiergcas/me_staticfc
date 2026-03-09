# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Generic Kernel (2025a)
#     language: python
#     name: generic_2025a
# ---

# # Figure 3
#
# This figure shows pBOLD for data expected to be dominated by So fluctuations (e.g., cardiac-gated data with minimal pre-processing) and BOLD fluctuations (low motion, constant-gated, fully pre-processed data)
#
# * We create a scatter plot comparing these two.
# * We create a distribution of trigger delay times for two representative scans

# +
import seaborn as sns
import pandas as pd
import panel as pn
import matplotlib.pyplot as plt
from utils.basics import get_dataset_index, DOWNLOAD_DIRS
import pickle
import os.path as osp

pn.extension()
# -

DATASET='discovery'
CENSOR_MODE='ALL'

with open(f'./cache/{DATASET}_QC_metrics_{CENSOR_MODE}.pkl', 'rb') as f:
    QC_metrics = pickle.load(f)
QC_metrics.keys()

pBOLD = QC_metrics['C','pBOLD'].set_index(['Subject','Session','Pre-processing','NORDIC'])

BOLD_heavy_data = pBOLD.loc[:,'constant_gated','ALL_Tedana-fastica','on']
So_heavy_data   = pBOLD.loc[:,'cardiac_gated','ALL_Basic','off']
df = pd.concat([BOLD_heavy_data,So_heavy_data],axis=1)
df.columns=['BOLD Dominated','So Dominated']

sns.set_context("paper", rc={"xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 16, 'legend.fontsize':16})
fig, axs = plt.subplots(1,1,figsize=(6,6));
sns.despine(top=True, right=True)
sns.set_palette('Set2')
f = sns.barplot(data=df)
sns.swarmplot(data=df, edgecolor="black", linewidth=0.5, alpha=0.7,s=8)
sns.despine()
f.set_ylim(0,1)
f.grid(False)
f.set_ylabel('$p_{BOLD}$')

pn.Row(pn.Column(pn.pane.Markdown('### Cardiac Gated'),So_heavy_data.sort_values(by='pBOLD')),
       pn.Column(pn.pane.Markdown('### Constant Gated'),BOLD_heavy_data.sort_values(by='pBOLD')))

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

# # Examine Triggers for cardiac-gated scans

ds_index = get_dataset_index('discovery')

df = None
for sbj,ses in ds_index:
    if ses == 'constant_gated':
        continue
    path = osp.join(DOWNLOAD_DIRS['discovery'],sbj,ses,'func',f'{sbj}_{ses}_task-rest.Triggers.1D')
    try:
        trigger_df = pd.read_csv(path, sep=' ', skiprows=2, header=None)
        trigger_df.columns = ['onset','slice','acquisition']
        trigger_df = trigger_df.set_index(['slice','acquisition']).loc[0,:].diff()
        if df is None:
            df = trigger_df
        else:
            df = pd.concat([df,trigger_df])
    except:
        print('No trigger file available for %s' % path)

df.describe()

# Load Trigger info for cardiac-gated scan with highest pBOLD
path = '/data/SFIMJGC_HCP7T/BCBL2024/openeuro/meica_eval/MGSBJ02/cardiac_gated/func/MGSBJ02_cardiac_gated_task-rest.Triggers.1D'
trigger_df = pd.read_csv(path, sep=' ', skiprows=2, header=None)
trigger_df.columns = ['onset','slice','acquisition']
highest_pBOLD_tigger_onsets = trigger_df.set_index(['slice','acquisition']).loc[0,:].diff() - 2500

# Load Trigger info for cardiac gated scan with lowest pBOLD
path = '/data/SFIMJGC_HCP7T/BCBL2024/openeuro/meica_eval/MGSBJ05/cardiac_gated/func/MGSBJ05_cardiac_gated_task-rest.Triggers.1D'
trigger_df = pd.read_csv(path, sep=' ', skiprows=2, header=None)
trigger_df.columns = ['onset','slice','acquisition']
lowest_pBOLD_tigger_onsets = trigger_df.set_index(['slice','acquisition']).loc[0,:].diff() - 2500

df = pd.concat([lowest_pBOLD_tigger_onsets,highest_pBOLD_tigger_onsets,],axis=1).dropna()
df.columns = ['pBOLD=0.04','pBOLD=0.84']

sns.set(font_scale=1.3)
sns.set_style('white')
f = sns.histplot(data=df, palette='Oranges', bins=30, alpha=.8)
f.set_xlabel('TR Offset [ms]')
f.set_ylabel('Number of Acquisitions')
sns.despine()
