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
from itertools import combinations_with_replacement, combinations

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

# # 2. Load FC for each scan

fc_all_sbjs_R = {}
fc_all_sbjs_Z = {}

for pipeline in ['Basic','GSasis','Tedana','TedanaGSasis']:
    for censor_mode in ['ALL','KILL']:
        for e in ['e01','e02','e03']:
            fc_all_sbjs_R[f'{e}-{pipeline}-{censor_mode}']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
            fc_all_sbjs_Z[f'{e}-{pipeline}-{censor_mode}']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
            for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list, desc='%s,%s,%s' % (pipeline,censor_mode,e))):
                netts_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netts');
                netts      = np.loadtxt(netts_path)
                netcc      = pd.DataFrame(netts).corr()
                fc_all_sbjs_R[f'{e}-{pipeline}-{censor_mode}'].loc[(sbj,ses),:] = sym_matrix_to_vec(netcc.values, discard_diagonal=True)
                fc_all_sbjs_Z[f'{e}-{pipeline}-{censor_mode}'].loc[(sbj,ses),:] = np.arctanh(fc_all_sbjs_R[f'{e}-{pipeline}-{censor_mode}'].loc[(sbj,ses),:])

# ## 2.4. Projection across echoes (equivalent to taking the mean)
#
# First we will extract the timeseries with nilarn becuase netts do not have great precision

echo_pairs_tuples = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs        = [('|').join(i) for i in echo_pairs_tuples]

for pipeline in ['Basic','GSasis','Tedana','TedanaGSasis']:
    for censor_mode in ['ALL','KILL']:
        fc_all_sbjs_R[f'across_echoes_{pipeline}-{censor_mode}']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
        fc_all_sbjs_Z[f'across_echoes_{pipeline}-{censor_mode}']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)

        for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list,desc='%s,%s' % (pipeline,censor_mode))):
            aux_fc_xr = xr.DataArray(dims=['pair','edge'], coords={'pair':echo_pairs, 'edge':np.arange(Ncons)})
            
            for (e_x,e_y) in echo_pairs_tuples:
                roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netts')
                roi_ts_x      = np.loadtxt(roi_ts_path_x)
                roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netts')
                roi_ts_y      = np.loadtxt(roi_ts_path_y)
                aux_ts_x      = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
                aux_ts_y      = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
                # Compute the full correlation matrix between aux_ts_x and aux_ts_y
                aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
                aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
                aux_fc_xr.loc['|'.join((e_x,e_y)),:] = np.arctanh(aux_r_v)



# +
xr_basic = xr.DataArray(dims=['scan','echo_pairing','statistic'],
                        coords={'scan':['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list],
                                'echo_pairing':['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)],
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
        aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = aux_r_v

    for pair_of_pairs in combinations(fc_xr_all.pair.values,2):
        slope, intercept = np.polyfit(fc_xr_all.sel(pair=pair_of_pairs[0]),fc_xr_all.sel(pair=pair_of_pairs[1]),1)
        xr_basic.loc['_'.join((sbj,ses)),'_vs_'.join(pair_of_pairs),'Slope'] = slope
        xr_basic.loc['_'.join((sbj,ses)),'_vs_'.join(pair_of_pairs),'Intercept'] = intercept

# +
xr_meica = xr.DataArray(dims=['scan','echo_pairing','statistic'],
                        coords={'scan':['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list],
                                'echo_pairing':['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)],
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
        aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = aux_r_v

    for pair_of_pairs in combinations(fc_xr_all.pair.values,2):
        slope, intercept = np.polyfit(fc_xr_all.sel(pair=pair_of_pairs[0]),fc_xr_all.sel(pair=pair_of_pairs[1]),1)
        xr_meica.loc['_'.join((sbj,ses)),'_vs_'.join(pair_of_pairs),'Slope'] = slope
        xr_meica.loc['_'.join((sbj,ses)),'_vs_'.join(pair_of_pairs),'Intercept'] = intercept
# -

slope_plot = xr_basic.sel(statistic='Slope').to_dataframe('Slope').hvplot.kde(label='Basic Denosing') * \
xr_meica.sel(statistic='Slope').to_dataframe('Slope').hvplot.kde(label='MEICA Denoising') * \
hv.VLine(1).opts(color='k')

intercept_plot = xr_basic.sel(statistic='Intercept').to_dataframe('Intercept').hvplot.kde(label='Basic Denoising') * \
xr_meica.sel(statistic='Intercept').to_dataframe('Intercept').hvplot.kde(label='MEICA Denoising') * \
hv.VLine(0).opts(color='k')

slope_plot + intercept_plot

xr_basic_scan_avg = xr_basic.mean(dim='echo_pairing')
xr_meica_scan_avg = xr_meica.mean(dim='echo_pairing')

slope_plot = xr_basic_scan_avg.sel(statistic='Slope').to_dataframe('Slope').hvplot.kde(label='Basic Denosing') * \
xr_meica_scan_avg.sel(statistic='Slope').to_dataframe('Slope').hvplot.kde(label='MEICA Denoising') * \
hv.VLine(1).opts(color='k')
intercept_plot = xr_basic_scan_avg.sel(statistic='Intercept').to_dataframe('Intercept').hvplot.kde(label='Basic Denoising') * \
xr_meica_scan_avg.sel(statistic='Intercept').to_dataframe('Intercept').hvplot.kde(label='MEICA Denoising') * \
hv.VLine(0).opts(color='k')

slope_plot + intercept_plot

pd.DataFrame(xr_basic_scan_avg.values,index=xr_basic_scan_avg.scan,columns=xr_basic_scan_avg.statistic).reset_index().hvplot.scatter(x='Intercept',y='Slope',hover_cols='index',aspect='square',xlim=(-0.02,0.105),ylim=(0.4,1.1)) * \
hv.VLine(0).opts(color='k',line_dash='dashed',line_width=1) * hv.HLine(1).opts(color='k',line_dash='dashed',line_width=1)

df = pd.DataFrame([xr_basic_scan_avg.sel(statistic='Slope').values,xr_meica_scan_avg.sel(statistic='Slope').values]).T
df.columns=['Basic Denosing','MEICA Denosing']
df.index=xr_basic_scan_avg.scan
df.index.name='Scan'
df.name = 'Slope'
df.T.hvplot()

# ## 2.5. Projection across echoes (following MEICA denoising)

# +
fc_all_sbjs_R['across_echoes_MEICA-ALL']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
fc_all_sbjs_Z['across_echoes_MEICA-ALL']  = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
fc_all_sbjs_R['across_echoes_MEICA-KILL'] = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
fc_all_sbjs_Z['across_echoes_MEICA-KILL'] = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
fc_all_sbjs_R['across_echoes_MEICA-CENSOR'] = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)
fc_all_sbjs_Z['across_echoes_MEICA-CENSOR'] = pd.DataFrame(np.ones((Nscans,Ncons))* np.nan, index=dataset_info_df.index)

for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    fc_xr_all       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    fc_xr_kill       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    
    fc_xr_censor       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    censor_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'motion_{sbj}_censor.1D')
    censor     = np.loadtxt(censor_path).astype(bool)
    
    for (e_x,e_y) in echo_pairs_tuples:
        roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.meica_dn.scale.tproject_ALL.{ATLAS_NAME}_000.netts')
        roi_ts_x      = np.loadtxt(roi_ts_path_x)
        roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.meica_dn.scale.tproject_ALL.{ATLAS_NAME}_000.netts')
        roi_ts_y      = np.loadtxt(roi_ts_path_y)
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = np.arctanh(aux_r_v)

        roi_ts_x = roi_ts_x[censor,:]
        roi_ts_y = roi_ts_y[censor,:]
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        fc_xr_censor.loc['|'.join((e_x,e_y)),:] = np.arctanh(aux_r_v)

        roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.meica_dn.scale.tproject_KILL.{ATLAS_NAME}_000.netts')
        roi_ts_x      = np.loadtxt(roi_ts_path_x)
        roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.meica_dn.scale.tproject_KILL.{ATLAS_NAME}_000.netts')
        roi_ts_y      = np.loadtxt(roi_ts_path_y)
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        fc_xr_kill.loc['|'.join((e_x,e_y)),:] = np.arctanh(aux_r_v)
        
    fc_all_sbjs_Z['across_echoes_MEICA-ALL'].loc[(sbj,ses),:] = fc_xr_all.mean(axis=0).values
    fc_all_sbjs_R['across_echoes_MEICA-ALL'].loc[(sbj,ses),:] = np.tanh(fc_xr_all.mean(axis=0).values)
    fc_all_sbjs_Z['across_echoes_MEICA-KILL'].loc[(sbj,ses),:] = fc_xr_kill.mean(axis=0).values
    fc_all_sbjs_R['across_echoes_MEICA-KILL'].loc[(sbj,ses),:] = np.tanh(fc_xr_kill.mean(axis=0).values)
    fc_all_sbjs_Z['across_echoes_MEICA-CENSOR'].loc[(sbj,ses),:] = fc_xr_censor.mean(axis=0).values
    fc_all_sbjs_R['across_echoes_MEICA-CENSOR'].loc[(sbj,ses),:] = np.tanh(fc_xr_censor.mean(axis=0).values)
# -

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

# + active=""
# high_mot_scans = list(mot_mean_fd_DF[mot_mean_fd_DF['Mean FD'] > mot_mean_fd_DF.median().values[0]].index)
# low_mot_scans  = list(mot_mean_fd_DF[mot_mean_fd_DF['Mean FD'] < mot_mean_fd_DF.median().values[0]].index)
# print(len(high_mot_scans),len(low_mot_scans))
# -

# ***
# # 4. Correlation between motion and FC variability across scans

mot_vs_fc = {}
scenarios =  ['E01-volreg-ALL','E02-volreg-ALL','E03-volreg-ALL',
              'E01-volreg-KILL','E02-volreg-KILL','E03-volreg-KILL',
              'E01-volreg-CENSOR','E02-volreg-CENSOR','E03-volreg-CENSOR',
              'E01-MEICA-ALL','E02-MEICA-ALL','E03-MEICA-ALL',
              'E01-MEICA-KILL','E02-MEICA-KILL','E03-MEICA-KILL',
              'E01-MEICA-CENSOR','E02-MEICA-CENSOR','E03-MEICA-CENSOR',
              'OC-ALL','OC-KILL',None,
              'MEICA-ALL','MEICA-KILL','MEICA-CENSOR',
              'across_echoes_volreg-ALL','across_echoes_volreg-KILL','across_echoes_volreg-CENSOR',
              'across_echoes_MEICA-ALL','across_echoes_MEICA-KILL','across_echoes_MEICA-CENSOR']

# ## 4.1. Following Motion Correction

for s,scenario in enumerate(scenarios):
    if scenario is None:
        continue
    aux = []
    for c in tqdm(np.arange(Ncons), desc='[%d/%d] %s' % (s+1,len(scenarios),scenario)):
        cc = np.corrcoef(fc_all_sbjs_Z[scenario].loc[:,c].values,mot_mean_fd_DF['Mean FD'].values)[0,1]
        aux.append(cc)
    mot_vs_fc[scenario]  = np.array(aux)

# ## 4.5. Plot the relationships

plot_grid = pn.layout.GridBox(ncols=3)
dfs       = {}
#scenarios_kill = [s for s in scenarios if 'MEICA-KILL' in s]
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

plots = {}
dfs       = {}
#scenarios_kill = [s for s in scenarios if 'MEICA-KILL' in s]
for scenario in tqdm(scenarios, desc='All scenarios'):
    if scenario is None:
        continue
    dfs[scenario]           = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc[scenario]]).T,           columns=['Distance [mm]','QC:RSFC r'])
    plots[scenario] = dfs[scenario].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r', line_width=1, ylim=(-0.5,0.5), label=scenario).opts(title=scenario) * \
           hv.HLine(0).opts(line_color='gray', line_width=0.5)

(plots['E02-volreg-KILL'] * plots['E02-MEICA-KILL'] * plots['MEICA-KILL']* plots['across_echoes_MEICA-KILL']* plots['across_echoes_volreg-KILL']).opts(width=300, legend_position='top')

(plots['E02-volreg-ALL'] * plots['E02-MEICA-ALL'] * plots['MEICA-ALL']* plots['across_echoes_MEICA-ALL'] * plots['across_echoes_volreg-ALL']).opts(width=300, legend_position='top')

(plots['E02-volreg-CENSOR'] * plots['E02-MEICA-CENSOR'] * plots['MEICA-CENSOR']* plots['across_echoes_MEICA-CENSOR'] * plots['across_echoes_volreg-CENSOR']).opts(width=300, legend_position='top')

plot = dfs['MEICA-KILL'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='r', line_width=3, ylim=(-0.3,0.3)).opts(title=scenario) * \
dfs['across_echoes_MEICA-KILL'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='g', line_width=3, line_dash='dashed', ylim=(-0.3,0.3)).opts(title=scenario) * \
dfs['E01-MEICA-KILL'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='b', line_width=3, ylim=(-0.3,0.3)).opts(title=scenario) * \
dfs['E02-MEICA-KILL'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='c', line_width=3, ylim=(-0.3,0.3)).opts(title=scenario) * \
dfs['E03-MEICA-KILL'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='y', line_width=3, ylim=(-0.3,0.3)).opts(title=scenario) * \
((dfs['E01-MEICA-KILL']+dfs['E02-MEICA-KILL']+dfs['E03-MEICA-KILL'])/3).sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='pink', line_width=3, ylim=(-0.3,0.3)).opts(title=scenario) * \
           hv.HLine(0).opts(line_color='gray', line_width=0.5)
plot



# +
plot_volreg = dfs['E02-volreg'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['E02-volreg'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='E02 Volreg') * \
hv.HLine(0).opts(line_color='gray')

plot_oc = dfs['OC'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['OC'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='OC') * \
hv.HLine(0).opts(line_color='gray')

plot_ocscrub = dfs['OC_Scrubbing'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['OC_Scrubbing'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='OC + Scrubbing') * \
hv.HLine(0).opts(line_color='gray')

plot_meica = dfs['MEICA'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['MEICA'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='MEICA Denoised') * \
hv.HLine(0).opts(line_color='gray')

plot_across_echoes_volreg = dfs['across_echoes_volreg'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['across_echoes_volreg'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='Across Echoes (after volreg)') * \
hv.HLine(0).opts(line_color='gray')

plot_across_echoes_MEICA = dfs['across_echoes_MEICA'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['across_echoes_MEICA'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='Across Echoes (after MEICA)') * \
hv.HLine(0).opts(line_color='gray')
# -

plot_volreg + plot_oc + plot_ocscrub + plot_meica + plot_across_echoes_volreg + plot_across_echoes_MEICA







mot_vs_fc_HL = {}

aux_high,aux_low = [],[]
aux_OC_high = fc_all_sbjs_Z['OC'].loc[high_mot_scans,:]
aux_OC_low  = fc_all_sbjs_Z['OC'].loc[low_mot_scans,:]
aux_mot_high = mot_mean_fd_DF.loc[high_mot_scans,'Mean FD'].values
aux_mot_low  = mot_mean_fd_DF.loc[high_mot_scans,'Mean FD'].values
for c in tqdm(np.arange(Ncons)):
    cc = np.corrcoef(aux_OC_high.loc[:,c].values,aux_mot_high)[0,1]
    aux_high.append(cc)
    cc = np.corrcoef(aux_OC_low.loc[:,c].values,aux_mot_low)[0,1]
    aux_low.append(cc)
mot_vs_fc_HL['OC']  = np.array(aux_high) - np.array(aux_low)

mot_vs_fc_scrubbing = {}

aux = []
for c in tqdm(np.arange(Ncons)):
    cc_orig      = np.corrcoef(fc_all_sbjs_Z['OC'].loc[:,c].values,mot_mean_fd_DF['Mean FD'].values)[0,1]
    cc_scrubbing = np.corrcoef(fc_all_sbjs_Z['OC_Scrubbing'].loc[:,c].values,mot_mean_fd_DF['Mean FD'].values)[0,1]
    aux.append(cc_scrubbing - cc_orig)
mot_vs_fc_scrubbing['OC']  = np.array(aux)









dfs = {}

dfs['E02-volreg']    = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc['E02-volreg']]).T, columns=['Distance [mm]','QC:RSFC r'])
dfs['OC']            = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc['OC']]).T, columns=['Distance [mm]','QC:RSFC r'])
dfs['MEICA'] = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc['MEICA']]).T, columns=['Distance [mm]','QC:RSFC r'])

plot_volreg = dfs['E02-volreg'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['E02-volreg'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='E02 Volreg') * \
hv.HLine(0).opts(line_color='gray')

plot_oc = dfs['OC'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['OC'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='OC') * \
hv.HLine(0).opts(line_color='gray')

plot_meicaregress = dfs['MEICA'].hvplot.scatter(x='Distance [mm]',y='QC:RSFC r', c='r',s=1, width=300) * \
dfs['MEICA'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='MEICA Denoised') * \
hv.HLine(0).opts(line_color='gray')

plot_volreg + plot_oc + plot_meicaregress

dfs['HL OC'] = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc_HL['OC']]).T, columns=['Distance [mm]','High-low motion dR'])

dfs['HL OC'].hvplot.scatter(x='Distance [mm]',y='High-low motion dR', c='r',s=1, width=300) * \
dfs['HL OC'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='High-low motion dR',c='w', line_width=3, ylim=(-0.5,0.5)).opts(title='OC') * \
hv.HLine(0).opts(line_color='gray')

dfs['Scrubbing OC'] = pd.DataFrame(np.vstack([roi_distance_vect,mot_vs_fc_scrubbing['OC']]).T, columns=['Distance [mm]','Scrubbing dR'])

dfs['Scrubbing OC'].hvplot.scatter(x='Distance [mm]',y='Scrubbing dR', c='r',s=1, width=300) * \
dfs['Scrubbing OC'].sort_values(by='Distance [mm]').rolling(400, center=True).mean().dropna().hvplot(x='Distance [mm]',y='Scrubbing dR',c='w', line_width=3, ylim=(-0.05,0.05)).opts(title='OC') * \
hv.HLine(0).opts(line_color='gray')

dfs['E02-volreg'].sort_values(by='Distance [mm]').rolling(500, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='r', line_width=3, ylim=(-0.5,0.5), width=300) * \
dfs['MEICA'].sort_values(by='Distance [mm]').rolling(500, center=True).mean().dropna().hvplot(x='Distance [mm]',y='QC:RSFC r',c='g', line_width=3, ylim=(-0.5,0.5))


