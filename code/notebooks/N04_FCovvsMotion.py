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
# One additional idea surrounding this project is that of using the projection to the mean (or average across all possible combinations) as a way to denoise the data. To evaluate the validity of this approach, we thought about three tests: 
#
# 1. Improvements in CPM modeling
# 2. Reduction of FC dependence with ROI distance
# 3. Improved test-retest reliability
#
# This notebook looks at item 2 using covariance as our estimate of functional connectivity. Our approach here follows that of Powers' et al. in ["Ridding fMRI dta of motion-related influences"](https://www.pnas.org/doi/10.1073/pnas.1720985115?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed)
#
#
# ***

import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import subprocess
import datetime
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, PRJ_DIR, CODE_DIR
ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
from nilearn.connectome import sym_matrix_to_vec
from sfim_lib.io.afni import load_netcc
import hvplot.pandas
import seaborn as sns
import holoviews as hv
import xarray as xr
import panel as pn
from itertools import combinations_with_replacement

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

# ### Compute Euclidean Distance between ROI centroids

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

# # 2. Load FC for all scans 
#
# Do this for all for all pipelines and censoring modes
#

fc_all_sbjs_Cov = {}

for pipeline in ['Basic','GSasis','Tedana','TedanaGSasis']:
    for censor_mode in ['ALL','KILL']:
        for e in ['e01','e02','e03']:
            fc_all_sbjs_Cov[f'{e}-{pipeline}-{censor_mode}']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)    
            for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list, desc='%s,%s,%s' %(pipeline,censor_mode,e))):
                netts_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netts');
                netts      = np.loadtxt(netts_path)
                netcc      = np.cov(netts.T,bias=True)
                fc_all_sbjs_Cov[f'{e}-{pipeline}-{censor_mode}'].loc[(sbj,ses),:] = sym_matrix_to_vec(netcc, discard_diagonal=True)

# ## 2.4. Projection across echoes (equivalent to taking the mean)
#
# First we will extract the timeseries with nilarn becuase netts do not have great precision

echo_pairs_tuples = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs        = [('|').join(i) for i in echo_pairs_tuples]

for pipeline in ['Basic','GSasis','Tedana','TedanaGSasis']:
    for censor_mode in ['ALL','KILL']:
        fc_all_sbjs_Cov[f'across_echoes_{pipeline}-{censor_mode}']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)

        for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list, desc='%s,%s' % (pipeline,censor_mode))):
            # Create empty XR to temporarily hold FC across all possible echo combinations
            aux_fc_xr  = xr.DataArray(dims=['pair','edge'],
                                      coords={'pair':  echo_pairs,
                                              'edge':  np.arange(Ncons)})
            # Fill up that temporary XR object
            for (e_x,e_y) in echo_pairs_tuples:
                roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netts')
                roi_ts_x      = np.loadtxt(roi_ts_path_x)
                roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netts')
                roi_ts_y      = np.loadtxt(roi_ts_path_y)
                aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
                aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
                # Compute the full correlation matrix between aux_ts_x and aux_ts_y
                aux_r   = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
                aux_fc_xr.loc['|'.join((e_x,e_y)),:] = sym_matrix_to_vec(aux_r, discard_diagonal=True)

            # Save the averate of all echo combinations on the final 
            fc_all_sbjs_Cov[f'across_echoes_{pipeline}-{censor_mode}'].loc[(sbj,ses),:] = aux_fc_xr.mean(axis=0).values

# # 3. Load Mean Motion Framewise Displacement
#
# According to [Power et al. "Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion"](https://www.sciencedirect.com/science/article/pii/S1053811911011815)
#
# #### Framewise displacement (FD) calculations
#
# "Differentiating head realignment parameters across frames yields a six dimensional timeseries that represents instantaneous head motion. To express instantaneous head motion as a scalar quantity we used the empirical formula, FDi = |Δdix| + |Δdiy| + |Δdiz| + |Δαi| + |Δβi| + |Δγi|, where Δdix = d(i − 1)x − dix, and similarly for the other rigid body parameters [dix diy diz αi βi γi]. Rotational displacements were converted from degrees to millimeters by calculating displacement on the surface of a sphere of radius 50 mm, which is approximately the mean distance from the cerebral cortex to the center of the head."

#mot_mean_fd = np.empty((Nscans,Ncons))
mot_mean_fd_DF = pd.DataFrame(index=dataset_info_df.index,columns=['Mean FD'])
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','dfile.r01.1D')
    # Load the motion parameters
    motion_params = np.loadtxt(mot_path)
    # Calculate the differences between consecutive time points
    diffs = np.diff(motion_params, axis=0)
    # Optionally convert rotational parameters (in radians) to mm (assuming a brain radius of 50 mm)
    brain_radius = 0.050  # in mm
    diffs[:, 3:] *= brain_radius
    # Compute framewise displacement (FD)
    FD = np.sum(np.abs(diffs), axis=1)
    # Pad with 0 for the first time point (no preceding frame to compare with)
    #FD = np.insert(FD, 0, 0)
    mot_mean_fd_DF.loc[(sbj,ses)] = FD.mean()
mot_mean_fd_DF = mot_mean_fd_DF.infer_objects()

# ***
# # 4. Correlation between motion and FC variability across scans

mot_vs_fc = {}
scenarios =  ['e02-Basic-ALL','e02-GSasis-ALL', 'e02-Tedana-ALL','across_echoes_Basic-ALL',
             'e02-Basic-KILL','e02-GSasis-KILL', 'e02-Tedana-KILL','across_echoes_Basic-KILL']

# Let's first estimate the correlation between scan motion and each edge FC across all echoes

for s,scenario in enumerate(scenarios):
    if scenario is None:
        continue
    aux = []
    for c in tqdm(np.arange(Ncons), desc='[%d/%d] %s' % (s+1,len(scenarios),scenario)):
        cc = np.corrcoef(fc_all_sbjs_Cov[scenario].loc[:,c].values,mot_mean_fd_DF['Mean FD'].values)[0,1]
        aux.append(cc)
    mot_vs_fc[scenario]  = np.array(aux)

# Let's now plot that information as in the original paper by Power's et al.

plot_grid = pn.layout.GridBox(ncols=4)
dfs       = {}
for scenario in tqdm(scenarios, desc='All scenarios'):
    if scenario is None:
        plot_grid.append(None)
        continue
    dfs[scenario]           = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc[scenario]]).T,           columns=['Distance [mm]','QC:RSFC r'])
    plot = dfs[scenario].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
           dfs[scenario].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title=scenario) * \
           hv.HLine(0).opts(line_color='gray', line_width=0.5)
    plot_grid.append(plot)

plot_grid
