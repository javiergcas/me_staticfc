# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: BOLD WAVES 2024a
#     language: python
#     name: bold_waves_2024a
# ---

# # Description
#
# This notebook will compute C-FC across all possible echo pairs both following Basic and Advanced denoising.
#
# It will then compute linear fits for the contrast between those.
#
# Finally, it generates summary figures of how the slope and intercept adheres to the situations when BOLD or non-BOLD alone dominate the data

import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import subprocess
import datetime
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, ATLAS_NAME, PRJ_DIR, CODE_DIR
ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
from nilearn.connectome import sym_matrix_to_vec
from sfim_lib.io.afni import load_netcc
import hvplot.pandas
import seaborn as sns
import holoviews as hv
import xarray as xr
import panel as pn
from itertools import combinations_with_replacement, combinations
import matplotlib.pyplot as plt

# # 1. Load Dataset Information

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
Nscans          = dataset_info_df.shape[0]
print('++ Number of scans: %s scans' % Nscans)
dataset_scan_list = list(dataset_info_df.index)
Nacqs = 201

# # 2. Load Atlas Information

# +
roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df.head(5)

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)

print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))
# -

# Compute Euclidean Distance between ROI centroids

# +
# Select the columns that correspond to position
roi_coords_df = roi_info_df.set_index(['ROI_Name'])[['pos_R','pos_A','pos_S']]

# Convert the DataFrame to a NumPy array
roi_coords = roi_coords_df.values

# Calculate the Euclidean distance using broadcasting
roi_distance_matrix = np.sqrt(((roi_coords[:, np.newaxis] - roi_coords) ** 2).sum(axis=2))

# Convert to DataFrame
roi_distance_df = pd.DataFrame(roi_distance_matrix, index=roi_coords_df.index, columns=roi_coords_df.index)
# -

roi_distance_vect = sym_matrix_to_vec(roi_distance_df.values, discard_diagonal=True)

# Create list of all echo combinations and combinations of those

echo_pairs_tuples   = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]
print('Echo Pairs[n=%d]=%s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d]=%s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

scan_names = ['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list]

echoes_dict = {'e01':13.7,'e02':30,'e03':47}
#echoes_dict = {'e01':14,'e02':29.96,'e03':45.92}
ideal_slopes = {}
for p in pairs_of_echo_pairs:
    x,y = p.split('_vs_')
    x_e1,x_e2 = x.split('|')
    y_e1,y_e2 = y.split('|')
    ideal_slopes[p] = (echoes_dict[y_e1] * echoes_dict[y_e2]) / (echoes_dict[x_e1] * echoes_dict[x_e2])
print(ideal_slopes)

# # 3. Load Basic Quality Information for each scan
#
# ## 3.1. Fraction of censored datapoints

mot_df = pd.DataFrame(index=scan_names,columns=['Percent Censored'])
mot_df.index.name = 'scan'
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    scan_name = '_'.join((sbj,ses))
    censor_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'motion_{sbj}_censor.1D')
    censor     = np.loadtxt(censor_path).astype(bool)
    mot_df.loc[scan_name,'Percent Censored'] = 100*(len(censor)-np.sum(censor))/len(censor)
    mot_df.loc[scan_name,'Percent Used']     = 100*(np.sum(censor))/len(censor)

# ## 3.2. Fraction of BOLD-like vs. non-BOLD like data

tedana_df = pd.DataFrame(index=scan_names,columns=['Var. likely-BOLD','Var. unlikely-BOLD','Var. low-variance'])
tedana_df.index.name = 'scan'
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    scan_name = '_'.join((sbj,ses))
    ica_table_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','tedana_r01','ica_metrics.tsv')
    ica_table = pd.read_csv(ica_table_path,sep='\t')
    tedana_df.loc[scan_name,'Var. likely-BOLD'] = ica_table.set_index(['classification_tags']).loc['Likely BOLD','variance explained'].sum()
    tedana_df.loc[scan_name,'Var. unlikely-BOLD'] = ica_table.set_index(['classification_tags']).loc['Unlikely BOLD','variance explained'].sum()
    tedana_df.loc[scan_name,'Var. accepted'] = ica_table.set_index(['classification']).loc['accepted','variance explained'].sum()
    tedana_df.loc[scan_name,'Var. rejected'] = ica_table.set_index(['classification']).loc['rejected','variance explained'].sum()

# ## 3.3. RMSE

rsme_df = pd.DataFrame(index=scan_names,columns=['Avg. RSME'])
rsme_df.index.name = 'scan'
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    scan_name = '_'.join((sbj,ses))
    rsme_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','tedana_r01','rmse.avg.txt')
    rsme = np.loadtxt(rsme_path)
    rsme_df.loc[scan_name,'Avg. RSME'] = rsme

# # 4. Pearson's FC Slope and Intercept
# ## 4.1. Load the data following basic denosing
#
# We will create an xr.DataArray that will hold the slope and intercept of contrasting all 15 FC matrices for each scan separately. 
#
# We wll also then compute the averages per scan, so that we can characterize a given scan in 2D space.

slope_inter_xr_all        = {}

# %%time
slope_inter_xr_all['Basic'] = xr.DataArray(dims=['scan','echo_pairing','statistic'],
                        coords={'scan':['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list],
                                'echo_pairing':pairs_of_echo_pairs,
                                'statistic':['Slope','Intercept']})
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    fc_xr_all       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    for (e_x,e_y) in echo_pairs_tuples:
        roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_ALL.{ATLAS_NAME}_000.netts')
        roi_ts_x      = np.loadtxt(roi_ts_path_x)
        roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_ALL.{ATLAS_NAME}_000.netts')
        roi_ts_y      = np.loadtxt(roi_ts_path_y)
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = aux_r_v

    for pair_of_pairs in pairs_of_echo_pairs:
        p1,p2=pair_of_pairs.split('_vs_')
        x = fc_xr_all.sel(pair=p1)
        y = fc_xr_all.sel(pair=p2)
        slope, intercept = np.polyfit(x,y,1)
        slope_inter_xr_all['Basic'].loc['_'.join((sbj,ses)),pair_of_pairs,'Slope'] = slope
        slope_inter_xr_all['Basic'].loc['_'.join((sbj,ses)),pair_of_pairs,'Intercept'] = intercept

# ## 4.2. Load data and compute slope and intercept following Advanced denoising

# %%time
slope_inter_xr_all['MEICA'] = xr.DataArray(dims=['scan','echo_pairing','statistic'],
                        coords={'scan':['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list],
                                'echo_pairing':pairs_of_echo_pairs,
                                'statistic':['Slope','Intercept']})
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    fc_xr_all       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    for (e_x,e_y) in echo_pairs_tuples:
        roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.meica_dn.scale.tproject_ALL.{ATLAS_NAME}_000.netts')
        roi_ts_x      = np.loadtxt(roi_ts_path_x)
        roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.meica_dn.scale.tproject_ALL.{ATLAS_NAME}_000.netts')
        roi_ts_y      = np.loadtxt(roi_ts_path_y)
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = aux_r_v

    for pair_of_pairs in pairs_of_echo_pairs:
        p1,p2=pair_of_pairs.split('_vs_')
        x = fc_xr_all.sel(pair=p1)
        y = fc_xr_all.sel(pair=p2)
        slope, intercept = np.polyfit(x,y,1)
        slope_inter_xr_all['MEICA'].loc['_'.join((sbj,ses)),pair_of_pairs,'Slope'] = slope
        slope_inter_xr_all['MEICA'].loc['_'.join((sbj,ses)),pair_of_pairs,'Intercept'] = intercept

# # 5. Compute DBOLD

# %%time
df = pd.DataFrame(index=scan_names,columns=['dist_BOLD_Basic','dist_BOLD_MEICA','dist_NonBOLD_Basic','dist_to_NonBOLD_MEICA'])
for scan_name in tqdm(scan_names):
    dist_BOLD_Basic, dist_NonBOLD_Basic = [],[]
    dist_BOLD_MEICA, dist_NonBOLD_MEICA = [],[]
    for pair_of_pairs in pairs_of_echo_pairs:
        BOLD_ideal_slope    = ideal_slopes[pair_of_pairs]
        NonBOLD_ideal_slope = 1
        dist_BOLD_Basic = dist_BOLD_Basic + [np.sqrt(((slope_inter_xr_all['Basic'].loc[scan_name,pair_of_pairs,'Intercept'].values-0)**2)+((slope_inter_xr_all['Basic'].loc[scan_name,pair_of_pairs,'Slope'].values-BOLD_ideal_slope)**2))]
        dist_BOLD_MEICA = dist_BOLD_MEICA + [np.sqrt(((slope_inter_xr_all['MEICA'].loc[scan_name,pair_of_pairs,'Intercept'].values-0)**2)+((slope_inter_xr_all['MEICA'].loc[scan_name,pair_of_pairs,'Slope'].values-BOLD_ideal_slope)**2))]
        dist_NonBOLD_Basic = dist_NonBOLD_Basic + [np.sqrt(((slope_inter_xr_all['Basic'].loc[scan_name,pair_of_pairs,'Intercept'].values-0)**2)+((slope_inter_xr_all['Basic'].loc[scan_name,pair_of_pairs,'Slope'].values-1)**2))]
        dist_NonBOLD_MEICA = dist_NonBOLD_MEICA + [np.sqrt(((slope_inter_xr_all['MEICA'].loc[scan_name,pair_of_pairs,'Intercept'].values-0)**2)+((slope_inter_xr_all['MEICA'].loc[scan_name,pair_of_pairs,'Slope'].values-1)**2))]
    df.loc[scan_name,'dist_BOLD_Basic'] = np.array(dist_BOLD_Basic).mean()
    df.loc[scan_name,'dist_BOLD_MEICA'] = np.array(dist_BOLD_MEICA).mean()
    df.loc[scan_name,'dist_NonBOLD_Basic'] = np.array(dist_NonBOLD_Basic).mean()
    df.loc[scan_name,'dist_NonBOLD_MEICA'] = np.array(dist_NonBOLD_MEICA).mean()
df.index.name='scan'

# Concatenate all data into a single datagrame for plotting

aux = pd.concat([df, mot_df, tedana_df,rsme_df],axis=1)
aux['Percent Censored'] = (aux['Percent Censored'].astype(float)+1)*2
aux['Percent Used'] = aux['Percent Used'].astype(float)
aux['Var. likely-BOLD'] = aux['Var. likely-BOLD'].astype(float)
aux['Var. unlikely-BOLD'] = aux['Var. unlikely-BOLD'].astype(float)
aux['Var. accepted'] = aux['Var. accepted'].astype(float)
aux['Var. rejected'] = aux['Var. rejected'].astype(float)
aux['Avg. RSME'] = aux['Avg. RSME'].astype(float)

aux.hvplot.scatter(x='dist_BOLD_Basic',   y='dist_BOLD_MEICA',    aspect='square', cmap='viridis', c='Var. accepted', hover_cols=['scan'], alpha=0.7,s='Percent Censored', xlim=(-.01,3.1), ylim=(-.01,3.1)).opts(clim=(0,30))* hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)+ \
aux.hvplot.scatter(x='dist_NonBOLD_Basic',y='dist_NonBOLD_MEICA', aspect='square', cmap='viridis', c='Var. accepted', hover_cols=['scan'], alpha=0.7,s='Percent Censored', xlim=(-.01,3.1), ylim=(-.01,3.1)).opts(clim=(0,30)) * hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)

aux.hvplot.scatter(x='dist_BOLD_Basic',   y='dist_BOLD_MEICA',    aspect='square', cmap='viridis', c='Avg. RSME', 
                   hover_cols=['scan'],s='Percent Censored', alpha=0.7, xlim=(-.01,3.1), ylim=(-.01,3.1),
                   xlabel='D_BOLD for Basic Denoising', ylabel='D_BOLD for Advanced Denosing').opts(fontscale=1.5)* hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)
