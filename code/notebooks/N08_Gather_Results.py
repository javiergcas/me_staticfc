#!/usr/bin/env python
# coding: utf-8

# # Description: Gather results and create final data structures
# 
# This notebook consolidates scan-level outputs produced by earlier processing notebooks into dataset-level summary artifacts under `code/notebooks/summary_files/`.
# The goal is to create canonical inputs for downstream QC, figure-generation, and CPM notebooks so they do not need to repeatedly recompute scan-level metrics.
# 
# If any upstream notebook regenerates FC, pBOLD, TSNR, GS, Tedana, ICA, or physiology outputs, rerun this notebook so downstream analyses read refreshed summary files.
# 
# ## Artifacts written by this notebook
# 
# 1. `{DATASET}_FC.pkl` (dict of FC/covariance matrices per scan, scenario, and echo pair)
# 2. `{DATASET}_pBOLD.nc` (xarray DataArray of pBOLD values)
# 3. `{DATASET}_TSNR.pkl` (dict of scan-level TSNR summary tables)
# 4. `{DATASET}_GS_info_and_ts.pkl` (dict of GS time series and GS summary metrics)
# 5. `{DATASET}_Tedana_QC.pkl` (dict of Tedana component-count / variance-explained tables)
# 6. `{DATASET}_ICAs.pkl` (dict of ICA time series and ICA summary metrics)
# 7. `{DATASET}_Physiological_Timeseries.pkl` (dict of cardiac and respiratory traces)
# 8. `{DATASET}_Physiological_Regressors.pkl` (dict of selected RVT/cardiac/respiratory regressors)
# 
# ## Summary File Dependency Map
# 
# | N08 artifact | Produced in section | Loaded by notebooks |
# |---|---|---|
# | `{DATASET}_FC.pkl` | Section 1 | `N09`, `N11`, `N13`, `N17` |
# | `{DATASET}_pBOLD.nc` | Section 2 | `N09`, `N10a`, `N10b`, `N11`, `N17` |
# | `{DATASET}_TSNR.pkl` | Section 3 | `N10a`, `N10b`, `N11`, `N17` |
# | `{DATASET}_GS_info_and_ts.pkl` | Section 4 | `N17` |
# | `{DATASET}_Tedana_QC.pkl` | Section 5.1 | `N17` |
# | `{DATASET}_ICAs.pkl` | Section 5.2 | `N17` |
# | `{DATASET}_Physiological_Timeseries.pkl` | Section 6.1 | `N11`, `N17` |
# | `{DATASET}_Physiological_Regressors.pkl` | Section 6.2 | `N17` |
# 

# In[106]:


import os.path as osp
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from utils.basics import ATLASES_DIR, CODE_DIR, PRCS_DATA_DIR, DOWNLOAD_DIRS, FMRI_TRS, FMRI_FINAL_NUM_SAMPLES, NUM_DISCARDED_VOLUMES, PIPELINES
from utils.basics import get_dataset_index, echo_pairs_tuples, get_altas_info, pairs_of_echo_pairs
from tqdm.notebook import tqdm


# In[74]:


from afnipy import lib_physio_reading as lpr
from afnipy import lib_physio_opts    as lpo
from afnipy.lib_afni1D import Afni1D
import copy


# ## Configuration: Select dataset
# 
# Choose `discovery` or `evaluation`. The selected dataset controls both the scan index and the filesystem roots used throughout the notebook.
# 

# In[75]:


DATASET = input ('Select Dataset (discovery or evaluation):')
DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]


# ## Configuration: Define preprocessing scenarios
# 
# `pp_opts` maps human-readable labels to canonical pipeline IDs used in filenames and dictionary/xarray keys.
# All downstream summary objects index data by these canonical IDs (e.g., `ALL_Basic`, `ALL_GS`, `ALL_Tedana-fastica`).
# 

# In[76]:


pp_opts = {'No Censoring | No Regression':'ALL_NoRegression',
           'No Censoring | Basic':'ALL_Basic',
           'No Censoring | GSR':'ALL_GS',
           'No Censoring | Tedana (fastica)':'ALL_Tedana-fastica'}


# ## Configuration: Atlas metadata
# 
# Build the dataset-specific atlas path (`Power264-{DATASET}`), then load ROI metadata.
# `roi_idxs` (MultiIndex over ROI name/id/hemisphere/network) is reused as the row/column index in FC matrices.
# 

# In[77]:


ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)


# In[78]:


roi_info_df, _ = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# ## Configuration: Enumerate scans
# 
# `ds_index` provides the `(Subject, Session)` combinations processed by all sections below.
# `ses_list` and `sbj_list` are later used as coordinate axes when initializing xarray objects.
# 

# In[79]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# ---
# # 1. Gather Functional Connectivity Estimates
# 
# This section loads ROI time series for each scan/scenario/echo-pair and computes:
# - Pearson correlation matrices (`'corr'`)
# - Covariance matrices (`'cov'`)
# 
# Both are stored in a dictionary keyed by scan + processing context.
# 

# In[44]:


fc_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_FC.pkl')
print('++ FC will be saved in %s' % fc_path)


# In[45]:


get_ipython().run_cell_magic('time', '', 'data_fc = {}\nprint(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for nordic in [\'off\',\'on\']:\n        for pp in pp_opts.values():\n            for (e_x,e_y) in echo_pairs_tuples:\n                d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                # Compose path to input TS\n                roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                # Load TS into memory\n                if (not osp.exists(roi_ts_path_x)) or (not osp.exists(roi_ts_path_y)):\n                    print(f\'++ WARNING: Missing input files for {sbj},{ses},{e_x},{e_y},{nordic},{pp}\')\n                    print(f\'            {roi_ts_path_x}\')\n                    print(f\'            {roi_ts_path_y}\')\n                    i+=1\n                    continue\n                roi_ts_x      = np.loadtxt(roi_ts_path_x)\n                roi_ts_y      = np.loadtxt(roi_ts_path_y)\n                aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df[\'ROI_Name\'].values)\n                aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df[\'ROI_Name\'].values)\n                # Compute the full correlation matrix between aux_ts_x and aux_ts_y\n                aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'corr\']  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)\n                data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'cov\']  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(fc_path, \'wb\') as f:\n    pickle.dump(data_fc, f)\nprint(\'[DONE]\')\n')


# ### Output Schema: `{DATASET}_FC.pkl`
# 
# - **Path**: `code/notebooks/summary_files/{DATASET}_FC.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (6-tuple)**:
#   - `(Subject, Session, Pre-processing, NORDIC, ee_vs_ee, fc_metric)`
#   - Example: `('sub-001', 'ses-1', 'ALL_Basic', 'off', 'e01|e02', 'corr')`
# - **Key domains**:
#   - `Subject`: from `ds_index`
#   - `Session`: from `ds_index`
#   - `Pre-processing`: values of `pp_opts`
#   - `m-NORDIC`: `{'off', 'on'}`
#   - `ee_vs_ee`: entries from `echo_pairs_tuples` joined with `'|'`
#   - `fc_metric`: `{'corr', 'cov'}` for correlation/covariance
# - **Dictionary value**: `pandas.DataFrame` of shape `(N_ROIs, N_ROIs)`
#   - `index` and `columns`: `roi_idxs` MultiIndex with levels `['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']`
# 
# ### Downstream notebooks that load this file
# - `N09_Figure03_pBOLDonDiscovery.ipynb`
# - `N11_Figure05_ScanLevel_ProblemIdentification.ipynb`
# - `N13_RunCPMAnalyses.ipynb`
# - `N17_Data_Exploration_Dashboard.ipynb`
# 

# ---
# # 2. Gather pBOLD estimates
# 
# This section assembles all pBOLD CSV outputs into a single 7D xarray object (`pBOLD_xr`) and saves it as NetCDF.
# 

# In[80]:


pBOLD_xr_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_pBOLD.nc')
print('++ pBOLD will be saved in %s' % pBOLD_xr_path)


# In[81]:


pBOLD_xr = xr.DataArray(dims=['sbj','ses','pp','nordic','fc_metric','ee_vs_ee','qc_metric',],
                         coords={'sbj':       sbj_list,
                                 'ses':       ses_list,
                                 'pp':        list(pp_opts.values()),
                                 'nordic':    ['off','on'],
                                 'fc_metric': ['corr','cov'],
                                 'ee_vs_ee':  pairs_of_echo_pairs+['scan'],
                                 'qc_metric': ['pBOLD']})


# In[82]:


get_ipython().run_cell_magic('time', '', 'print(f"++ INFO: Loading all pBOLD estimates for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for NORDIC in [\'off\',\'on\']:\n            for pp in pp_opts.values():\n                for fc_metric in [\'corr\',\'cov\']:\n                    pBOLD_path    = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-{NORDIC}\',f\'errts.{sbj}.r01.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.pBOLD_{fc_metric}.csv\')\n                    try:\n                        pBOLD = pd.read_csv(pBOLD_path)\n                    except:\n                        print(\'Problematic file: %s\' % pBOLD_path )\n                    pBOLD_xr.loc[sbj,ses,pp,NORDIC,fc_metric,pBOLD[\'ee_vs_ee\'].to_list()] = pBOLD[\'pBOLD\'].values.reshape(len(pairs_of_echo_pairs)+1,1)\n')


# In[83]:


get_ipython().run_cell_magic('time', '', "print('++ INFO: Saving to disk...',end='')\npBOLD_xr.to_netcdf(pBOLD_xr_path)\nprint('[DONE]')\n")


# Show sample level pBOLD values reported in manuscript table

# In[153]:


aux = pd.concat([pBOLD_xr.sel(ee_vs_ee='scan', fc_metric='cov',pp=PIPELINES[1:]).mean(dim=['sbj','ses']).to_dataframe(name='mean').drop(['fc_metric','ee_vs_ee'],axis=1).round(2),
           pBOLD_xr.sel(ee_vs_ee='scan', fc_metric='cov',pp=PIPELINES[1:]).std(dim=['sbj','ses']).to_dataframe(name='stdv').drop(['fc_metric','ee_vs_ee'],axis=1).round(2)],axis=1)
aux


# In[157]:


for pp in PIPELINES[1:]:
    this_increase = 100 * (aux.loc[(pp,'on'),'mean'] - aux.loc[(pp,'off'),'mean']) / aux.loc[(pp,'on'),'mean']
    print('++ %s (pBOLD off --> pBOLD on) increase of %.1f %%' % (pp, this_increase.iloc[0]))


# ### Output Schema: `{DATASET}_pBOLD.nc`
# 
# - **Path**: `code/notebooks/summary_files/{DATASET}_pBOLD.nc`
# - **Serialized object type**: `xarray.DataArray`
# - **Dimensions**:
#   - `['sbj', 'ses', 'pp', 'nordic', 'fc_metric', 'ee_vs_ee', 'qc_metric']`
# - **Coordinates**:
#   - `sbj`: `sbj_list`
#   - `ses`: `ses_list`
#   - `pp`: `list(pp_opts.values())`
#   - `nordic`: `['off', 'on']`
#   - `fc_metric`: `['corr', 'cov']`
#   - `ee_vs_ee`: `pairs_of_echo_pairs + ['scan']`
#   - `qc_metric`: `['pBOLD']`
# - **Stored value semantics**:
#   - Scalar pBOLD estimate for each coordinate combination.
# 
# ### Downstream notebooks that load this file
# 
# - `N09_Figure03_pBOLDonDiscovery.ipynb`
# - `N10a_TSNR_and_pBOLD_BarPlots_Figure04.evaluation.ipynb`
# - `N10b_TSNR_and_pBOLD_BarPlots_SuppFig03.discovery.ipynb`
# - `N11_Figure05_ScanLevel_ProblemIdentification.ipynb`
# - `N17_Data_Exploration_Dashboard.ipynb`
# 

# ---
# # 3. Gather TSNR estimates
# 
# This section collects TSNR summary values (full-brain and visual-cortex) into DataFrames, then stores them in a dictionary keyed by metric context.
# 

# In[111]:


TSNR_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_TSNR.pkl')
print('++ TSNR will be saved in %s' % TSNR_path)


# In[112]:


get_ipython().run_cell_magic('time', '', 'TSNR = {}\nprint(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor TSNR_metric in [\'TSNR (Full Brain)\',\'TSNR (Visual Cortex)\']:\n    aux_df = pd.DataFrame(columns=[\'Subject\',\'Session\',\'Pre-processing\',\'m-NORDIC\',TSNR_metric])\n    aux_df.set_index([\'Subject\',\'Session\',\'Pre-processing\',\'m-NORDIC\'], inplace=True)\n    for sbj,ses in tqdm(list(ds_index), desc=TSNR_metric):\n        partial_key = (sbj, ses)\n        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)\n        if not sbj_ses_in_fc:\n            print(\'++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.\' % (sbj,ses))\n            continue\n        for pp in pp_opts.values():\n            for nordic in [\'off\',\'on\']:            \n                d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                if TSNR_metric == \'TSNR (Visual Cortex)\':\n                    aux_rois_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,\'tsnr_stats_regress\',f\'TSNR_ROIs_e02_{pp}.txt\')\n                    aux_rois      = pd.read_csv(aux_rois_path,skiprows=3, sep=r\'\\s+\').drop(0).set_index(\'ROI_name\')\n                    aux_df.loc[sbj,ses,pp,nordic] = float(aux_rois.loc[\'GHCP-R_Primary_Visual_Cortex\',\'Tmed\'])\n                if TSNR_metric == \'TSNR (Full Brain)\':\n                    aux_fb_path   = osp.join(PRCS_DATA_DIR,sbj,d_folder,\'tsnr_stats_regress\',f\'TSNR_FB_e02_{pp}.txt\')\n                    aux_fb        = pd.read_csv(aux_fb_path,skiprows=3, sep=r\'\\s+\').drop(0).set_index(\'ROI_name\')\n                    aux_df.loc[sbj,ses,pp,nordic] = float(aux_fb.loc[\'NONE\',\'Tmed\'])\n    TSNR[\'cov\',TSNR_metric] = aux_df.reset_index()\n    TSNR[\'corr\',TSNR_metric] = aux_df.reset_index()\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(TSNR_path, \'wb\') as f:\n    pickle.dump(TSNR, f)\nprint(\'[DONE]\')\n')


# In[142]:


aux = pd.concat([TSNR[('cov','TSNR (Full Brain)')].groupby(['Pre-processing','m-NORDIC'])['TSNR (Full Brain)'].mean().round(0).loc[PIPELINES[1:],:],
TSNR[('cov','TSNR (Full Brain)')].groupby(['Pre-processing','m-NORDIC'])['TSNR (Full Brain)'].std().round(0).loc[PIPELINES[1:],:]],axis=1)
aux.columns = ['TSNR (mean)','TSNR (stdv)']
aux


# In[149]:


for pp in PIPELINES[1:]:
    this_increase = 100 * (aux.loc[(pp,'on'),'TSNR (mean)'] - aux.loc[(pp,'off'),'TSNR (mean)']) / aux.loc[(pp,'on'),'TSNR (mean)']
    print('++ %s (TSNR off --> TSNR on) increase of %.0f %%' % (pp, this_increase))


# ### Output Schema: `{DATASET}_TSNR.pkl`
# 
# - **Path**: `code/notebooks/summary_files/{DATASET}_TSNR.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (2-tuple)**:
#   - `(fc_metric, tsnr_metric)`
#   - `fc_metric` in `{'cov', 'corr'}`
#   - `tsnr_metric` in `{'TSNR (Full Brain)', 'TSNR (Visual Cortex)'}`
# - **Dictionary value**: `pandas.DataFrame`
#   - Columns: `['Subject', 'Session', 'Pre-processing', 'm-NORDIC', <metric column>]`
#   - `<metric column>` is either `TSNR (Full Brain)` or `TSNR (Visual Cortex)`
#   - One row per `(Subject, Session, Pre-processing, NORDIC)` available in the source data.
# - **Storage note**:
#   - The same table is written under both `('cov', tsnr_metric)` and `('corr', tsnr_metric)` for downstream convenience.
# 
# ### Downstream notebooks that load this file
# 
# - `N10a_TSNR_and_pBOLD_BarPlots_Figure04.evaluation.ipynb`
# - `N10b_TSNR_and_pBOLD_BarPlots_SuppFig03.discovery.ipynb`
# - `N11_Figure05_ScanLevel_ProblemIdentification.ipynb`
# - `N17_Data_Exploration_Dashboard.ipynb`
# 

# ***
# # 4. Gather Global Signal Timeseries
# 
# This section builds a dataset-level dictionary with:
# - GS time series in signal-percent-change units for `OC`, `TE1`, `TE2`, and `TE3`
# - Scan-level GS summary metrics (`Adj R2 Physio`, `kappa`, `rho`)
# 
# For the discovery dataset, both entries are written as `None` placeholders.

# In[52]:


if DATASET == 'evaluation':
    kappa_rho_df = pd.read_csv(f'./summary_files/{DATASET}_gs_kappa_rho.csv', index_col=[0,1])
    print("++ INFO: The shape of kappa_rho_df is %s" % str(kappa_rho_df.shape))
else:
    kappa_rho_df = None


# In[53]:


gs_sf_path = f'./summary_files/{DATASET}_GS_info_and_ts.pkl'
print('++ INFO: Global Signal will be saved to %s' % gs_sf_path)


# In[54]:


fMRI_Sampling_Rate      = FMRI_TRS[DATASET]
fMRI_Preproc_Nsamples   = FMRI_FINAL_NUM_SAMPLES[DATASET]
fMRI_Ndiscard_samples   = NUM_DISCARDED_VOLUMES[DATASET]
fMRI_Preproc_Start_Time = str(float(fMRI_Sampling_Rate.replace('s','')) * int(fMRI_Ndiscard_samples))+'s'
print('++ fMRI Sampling Rate            = %s' % fMRI_Sampling_Rate)
print('++ fMRI Final Number of Samples  = %d Acquisitions' % fMRI_Preproc_Nsamples)
print('++ fMRI Number discarded Samples = %d Acquisitions' % fMRI_Ndiscard_samples)
print('++ fMRI Preproc Start Time       = %s' % fMRI_Preproc_Start_Time)


# In[55]:


fMRI_Preproc_index = pd.timedelta_range(start=fMRI_Preproc_Start_Time, periods=fMRI_Preproc_Nsamples, freq=fMRI_Sampling_Rate)
fMRI_Preproc_index[0:3]


# In[56]:


gs_df_dict = {}
for sbj,ses in tqdm(ds_index):
    if DATASET == 'evaluation':
        # Load All the GS versions (each echo time and optimally combined)
        gs_e01_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e01.volreg.GS.1D')
        gs_e02_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.1D')
        gs_e03_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e03.volreg.GS.1D')
        gs_OC_path  = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb06.{sbj}.r01.tedana_fastica_OC.GS.1D')
        gs_e01 = np.loadtxt(gs_e01_path)
        gs_e02 = np.loadtxt(gs_e02_path)
        gs_e03 = np.loadtxt(gs_e03_path)
        gs_OC  = np.loadtxt(gs_OC_path)
        gs_df = pd.DataFrame([gs_OC,gs_e01,gs_e02,gs_e03],index=['OC','TE1','TE2','TE3']).T
        gs_df = gs_df.infer_objects()
        gs_df.index = fMRI_Preproc_index
        gs_df.index.name = 'Time'
        gs_df.columns.name='Echo Time'
        # Transform to units of Signal Percent Change
        gs_df_spc = 100*(gs_df-gs_df.mean())/gs_df.mean()
        gs_df_dict[(sbj,ses,'gs_ts')] = gs_df_spc

        # Create Dataframe with Gs properties of interest
        GS_phys_match_file = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
        try:
            with open(GS_phys_match_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            gs_adjr2_physio = float(loaded_dict['model'].rsquared_adj)
        except:
            gs_adjr2_physio = None
        gs_kappa        = float(kappa_rho_df.loc[(sbj,ses),'kappa (GS)'])
        gs_rho          = float(kappa_rho_df.loc[(sbj,ses),'rho (GS)'])
        gs_df_metrics   = pd.DataFrame([gs_adjr2_physio,gs_kappa,gs_rho],index=['Adj R2 Physio','kappa','rho'],columns=['GS']).T
        gs_df_dict[sbj,ses,'gs_metrics'] = gs_df_metrics
    else:
        gs_df_dict[(sbj,ses,'gs_ts')] = None
        gs_df_dict[sbj,ses,'gs_metrics'] = None


# In[57]:


with open(gs_sf_path, 'wb') as f:
    pickle.dump(gs_df_dict, f)


# ### Output Schema: `{DATASET}_GS_info_and_ts.pkl`
# 
# - **Path**: `./summary_files/{DATASET}_GS_info_and_ts.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (3-tuple)**:
#   - `(Subject, Session, gs_object)` where `gs_object` is `'gs_ts'` or `'gs_metrics'`
# - **Dictionary value**:
#   - `'gs_ts'`: evaluation only, `pandas.DataFrame` indexed by `TimedeltaIndex` named `'Time'`
#     - Columns: `['OC', 'TE1', 'TE2', 'TE3']`
#     - `columns.name`: `'Echo Time'`
#     - Stored values are in signal percent change.
#   - `'gs_metrics'`: evaluation only, one-row `pandas.DataFrame`
#     - Index: `['GS']`
#     - Columns: `['Adj R2 Physio', 'kappa', 'rho']`
#   - Discovery dataset: `None` for both entries.
# 
# ### Downstream notebooks that load this file
# 
# - `N17_Data_Exploration_Dashboard.ipynb`
# 
# ---
# # 5. Gather Tedana Outputs
# 
# ## 5.1. TEDANA Statistics
# This section summarizes TEDANA component counts and explained-variance metrics per scan/scenario.
# Non-TEDANA pipelines are explicitly populated with `NaN` for these fields.
# 

# In[58]:


Tedana_QC_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_Tedana_QC.pkl')
print('++ Tedana basic statistics will be saved in %s' % Tedana_QC_path)


# Create empty data structures

# In[59]:


get_ipython().run_cell_magic('time', '', 'Tedana_QC = {}\nprint(f"++ INFO: Loading all Tedana_QC for the {DATASET} into memory...")\nfor tedana_metric in [\'#ICs (All)\',\'#ICs (Likely BOLD)\',\'#ICs (Unlikely BOLD)\',\'Var. Exp. (Likely BOLD)\',\'Var. Exp. (Unlikely BOLD)\']:\n    aux_df = pd.DataFrame(columns=[\'Subject\',\'Session\',\'Pre-processing\',\'m-NORDIC\',tedana_metric])\n    aux_df.set_index([\'Subject\',\'Session\',\'Pre-processing\',\'m-NORDIC\'], inplace=True)\n    Tedana_QC[\'cov\',tedana_metric] = aux_df\n')


# In[60]:


get_ipython().run_cell_magic('time', '', 'print(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for nordic in [\'off\',\'on\']:\n        for pp in pp_opts.values():\n            if \'Tedana\' not in pp:\n                Tedana_QC[\'cov\',\'#ICs (All)\'].loc[sbj,ses,pp,nordic]                = np.nan\n                Tedana_QC[\'cov\',\'#ICs (Likely BOLD)\'].loc[sbj,ses,pp,nordic]        = np.nan\n                Tedana_QC[\'cov\',\'#ICs (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic]      = np.nan\n                Tedana_QC[\'cov\',\'Var. Exp. (Likely BOLD)\'].loc[sbj,ses,pp,nordic]   = np.nan\n                Tedana_QC[\'cov\',\'Var. Exp. (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic] = np.nan\n            else:\n                tedana_type = pp.split(\'ALL_Tedana-\')[1]\n                d_folder    = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                ica_metrics_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'tedana_{tedana_type}\',\'ica_metrics.tsv\')\n                ica_metrics              = pd.read_csv(ica_metrics_path, sep=\'\\t\').set_index(\'Component\')\n                likely_bold_components   = list(ica_metrics[ica_metrics[\'classification_tags\']==\'Likely BOLD\'].index)\n                unlikely_bold_components = list(ica_metrics[ica_metrics[\'classification_tags\']==\'Unlikely BOLD\'].index)\n                \n                Tedana_QC[\'cov\',\'#ICs (All)\'].loc[sbj,ses,pp,nordic]              = ica_metrics.shape[0]\n                Tedana_QC[\'cov\',\'#ICs (Likely BOLD)\'].loc[sbj,ses,pp,nordic]      = len(likely_bold_components)\n                Tedana_QC[\'cov\',\'#ICs (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic]    = len(unlikely_bold_components)\n                Tedana_QC[\'cov\',\'Var. Exp. (Likely BOLD)\'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[likely_bold_components,\'variance explained\'].sum().round(2)\n                Tedana_QC[\'cov\',\'Var. Exp. (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[unlikely_bold_components,\'variance explained\'].sum().round(2)\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(Tedana_QC_path, \'wb\') as f:\n    pickle.dump(Tedana_QC, f)\nprint(\'[DONE]\')\n')


# ### Output Schema: `{DATASET}_Tedana_QC.pkl`
# 
# - **Path**: `code/notebooks/summary_files/{DATASET}_Tedana_QC.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (2-tuple)**:
#   - `('cov', tedana_metric)`
#   - `tedana_metric` in:
#     - `'#ICs (All)'`
#     - `'#ICs (Likely BOLD)'`
#     - `'#ICs (Unlikely BOLD)'`
#     - `'Var. Exp. (Likely BOLD)'`
#     - `'Var. Exp. (Unlikely BOLD)'`
# - **Dictionary value**: `pandas.DataFrame`
#   - Columns: `['Subject', 'Session', 'Pre-processing', 'm-NORDIC', <tedana_metric>]`
#   - For non-TEDANA pipelines, the metric field is populated with `NaN` by design.
# - **Storage note**:
#   - `N08` writes only `('cov', tedana_metric)` entries. `N17` duplicates those tables to `('corr', tedana_metric)` after loading because these metrics are FC-independent.
# 
# ### Downstream notebooks that load this file
# 
# - `N17_Data_Exploration_Dashboard.ipynb`
# 

# ## 5.2. ICA Timeseries and Additional Statistics

# In[61]:


fMRI_Sampling_Rate      = FMRI_TRS[DATASET]
fMRI_Preproc_Nsamples   = FMRI_FINAL_NUM_SAMPLES[DATASET]
fMRI_Ndiscard_samples   = NUM_DISCARDED_VOLUMES[DATASET]
fMRI_Preproc_Start_Time = str(float(fMRI_Sampling_Rate.replace('s','')) * int(fMRI_Ndiscard_samples))+'s'
print('++ fMRI Sampling Rate            = %s' % fMRI_Sampling_Rate)
print('++ fMRI Final Number of Samples  = %d Acquisitions' % fMRI_Preproc_Nsamples)
print('++ fMRI Number discarded Samples = %d Acquisitions' % fMRI_Ndiscard_samples)
print('++ fMRI Preproc Start Time       = %s' % fMRI_Preproc_Start_Time)


# In[62]:


fMRI_Preproc_index = pd.timedelta_range(start=fMRI_Preproc_Start_Time, periods=fMRI_Preproc_Nsamples, freq=fMRI_Sampling_Rate)
fMRI_Preproc_index[0:3]


# In[63]:


ica_sf_ts_path = f'./summary_files/{DATASET}_ICAs.pkl'
print('++INFO: IC Timeseries and additional stats saved to %s' % ica_sf_ts_path)


# In[64]:


ica_dict = {}
for sbj,ses in tqdm(ds_index):
    # Load IC Timeseries
    ic_ts_path         = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off','tedana_fastica',f'ica_mixing.tsv')
    ic_ts              = pd.read_csv(ic_ts_path, sep='\t')
    ic_ts              = ic_ts.infer_objects()
    ic_ts.index        = fMRI_Preproc_index
    ic_ts.index.name   = 'Time'
    ic_ts.columns.name = 'Components'
    ica_dict[(sbj,ses,'ic_ts')] = ic_ts
    # Load IC Properties
    ic_metrics = pd.read_csv(osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off','tedana_fastica',f'ica_metrics.tsv'),sep='\t', index_col=0)
    ic_metrics.index.name='Name'
    ic_metrics               = ic_metrics.round(2)[['kappa','rho','variance explained','classification_tags']]
    if DATASET == 'evaluation':
        ic_metrics['corrwithGS'] = ic_ts.corrwith(gs_df_dict[sbj,ses,'gs_ts']['OC'])
        ic_metrics.columns = ['kappa','rho','varepx','label','R(ic,GS)']
    else:
        ic_metrics['corrwithGS'] = np.nan
        ic_metrics.columns = ['kappa','rho','varepx','label','R(ic,GS)']
    ica_dict[(sbj,ses,'ic_metrics')] = ic_metrics


# In[65]:


with open(ica_sf_ts_path, 'wb') as f:
    pickle.dump(ica_dict, f)


# ### Output Schema: `{DATASET}_ICAs.pkl`
# 
# - **Path**: `./summary_files/{DATASET}_ICAs.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (3-tuple)**:
#   - `(Subject, Session, ica_object)` where `ica_object` is `'ic_ts'` or `'ic_metrics'`
# - **Dictionary value**:
#   - `'ic_ts'`: `pandas.DataFrame`
#     - Index: `TimedeltaIndex` named `'Time'`
#     - `columns.name`: `'Components'`
#     - Each column contains the mixing time series for one ICA component.
#   - `'ic_metrics'`: `pandas.DataFrame` indexed by component name
#     - Columns: `['kappa', 'rho', 'varepx', 'label', 'R(ic,GS)']`
#     - `R(ic,GS)` is `NaN` for the discovery dataset.
# 
# ### Downstream notebooks that load this file
# 
# - `N17_Data_Exploration_Dashboard.ipynb`
# 
# ---
# # 6. Physiological Traces
# 
# ## 6.1. Gather Respiratory and Cardiac Recordings
# 
# This section builds a dictionary of per-scan cardiac and respiratory time series (evaluation dataset only, when source files exist).
# 

# In[67]:


physio_ts_sf_path = f'./summary_files/{DATASET}_Physiological_Timeseries.pkl'
print('++ INFO: Physiological Recordings will be saved to: %s' % physio_ts_sf_path)


# In[68]:


get_ipython().run_cell_magic('time', '', '\nphysio_dict = {}\nfor sbj,ses in tqdm(ds_index):\n    if DATASET == \'evaluation\':\n        slibase_file = osp.join(PRCS_DATA_DIR,sbj,\'D06_Physio\',f\'{sbj}_{ses}_task-rest_echo-1_slibase.1D\')\n        if osp.exists(slibase_file):\n            # Load Physio by creating Afni RetroObj\n            phys_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.tsv.gz\')\n            json_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.json\')\n            dset_epi  = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_echo-1_bold.nii.gz\')\n            input_line = [\'./physio_calc.py\', \'-phys_file\', phys_file, \'-phys_json\', json_file, \'-dset_epi\', dset_epi, \n                            \'-prefilt_mode\', \'median\', \'-prefilt_max_freq\', \'50\', \'-verb\',\'0\']\n            args_orig  = copy.deepcopy(input_line)\n            args_dict  = lpo.main_option_processing( input_line )\n            retobj     = lpr.retro_obj( args_dict, args_orig=args_orig )\n            physio_start_time = retobj.start_time\n            # Extract Cardiac Timseries\n            card_end_time  = retobj.data[\'card\'].end_time\n            card_nsamples  = retobj.data[\'card\'].ts_orig.shape[0]\n            card_samp_rate = retobj.data[\'card\'].samp_rate\n            card_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=card_nsamples, freq=str(card_samp_rate)+"s")\n            card_df        = pd.DataFrame(retobj.data[\'card\'].ts_orig, index=card_index,columns=[\'PGG\'])\n            card_df.index.name = \'Time\'\n\n            # Extract Respiratory Timeseries\n            resp_end_time  = retobj.data[\'resp\'].end_time\n            resp_nsamples  = retobj.data[\'resp\'].ts_orig.shape[0]\n            resp_samp_rate = retobj.data[\'resp\'].samp_rate\n            resp_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=resp_nsamples, freq=str(resp_samp_rate)+"s")\n            resp_df        = pd.DataFrame(retobj.data[\'resp\'].ts_orig, index=resp_index,columns=[\'Respiration\'])\n            resp_df.index.name = \'Time\'\n\n            #Add to final dictionary\n            physio_dict[(sbj,ses,\'card\')] = card_df\n            physio_dict[(sbj,ses,\'resp\')] = resp_df\n        else:\n            physio_dict[(sbj,ses,\'card\')] = None\n            physio_dict[(sbj,ses,\'resp\')] = None\n    else:\n        physio_dict[(sbj,ses,\'card\')] = None\n        physio_dict[(sbj,ses,\'resp\')] = None\n')


# In[69]:


with open(physio_ts_sf_path, 'wb') as f:
    pickle.dump(physio_dict, f)


# ### Output Schema: `{DATASET}_Physiological_Timeseries.pkl`
# 
# - **Path**: `./summary_files/{DATASET}_Physiological_Timeseries.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (3-tuple)**:
#   - `(Subject, Session, signal_type)` where `signal_type` is `'card'` or `'resp'`
# - **Dictionary value**:
#   - For available evaluation scans:
#     - `'card'`: `pandas.DataFrame` with one column `['PGG']` and `TimedeltaIndex` named `'Time'`
#     - `'resp'`: `pandas.DataFrame` with one column `['Respiration']` and `TimedeltaIndex` named `'Time'`
#   - Missing/unavailable scans (or discovery dataset): `None`
# 
# ### Downstream notebooks that expect this file
# 
# - `N11_Figure05_ScanLevel_ProblemIdentification.ipynb` loads `./summary_files/{DATASET}_Physiological_Timeseries.pkl` with `pickle.load`.
# - `N17_Data_Exploration_Dashboard.ipynb` loads `./summary_files/{DATASET}_Physiological_Timeseries.pkl` with `pickle.load`.
# 
# ## 6.2. Gather Physiological Regressors
# 

# In[70]:


physio_regressors_sf_path = f'./summary_files/{DATASET}_Physiological_Regressors.pkl'
print('++ INFO: Physiological Regressors will be saved to %s' % physio_regressors_sf_path)


# In[71]:


physio_reg_dict = {}
for sbj,ses in tqdm(ds_index):
    if DATASET == 'evaluation':
        slibase_path = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio',f'{sbj}_{ses}_task-rest_echo-1_slibase.1D')
        if osp.exists(slibase_path):
            slibase_obj  = Afni1D(slibase_path)
            slibase_df   = pd.read_csv(slibase_path, comment='#', delimiter=' +', header=None, engine='python')
            slibase_df.columns=slibase_obj.labels
            slibase_df=slibase_df[3::].reset_index(drop=True)
            slibase_df.index = fMRI_Preproc_index
            slibase_df.index.name = 'Time'
            slibase_df.columns.name = 'Regressors'
            # Load list of selected regressors for the GS
            GS_phys_match_file = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
            with open(GS_phys_match_file, 'rb') as f:
                loaded_dict = pickle.load(f)
                selected_physio_regs = loaded_dict['selected_regs']
            selected_RVT_regs  = [r for r in selected_physio_regs if 'rvt' in r]
            selected_card_regs = [r for r in selected_physio_regs if 'card' in r]
            selected_resp_regs = [r for r in selected_physio_regs if 'resp' in r]
            physio_reg_dict[(sbj,ses,'RVT_regs')]  = slibase_df[selected_RVT_regs]
            physio_reg_dict[(sbj,ses,'card_regs')] = slibase_df[selected_card_regs]
            physio_reg_dict[(sbj,ses,'resp_regs')] = slibase_df[selected_resp_regs]
        else:
            physio_reg_dict[(sbj,ses,'RVT_regs')] = None
            physio_reg_dict[(sbj,ses,'card_regs')] = None
            physio_reg_dict[(sbj,ses,'resp_regs')] = None
    else:
        physio_reg_dict[(sbj,ses,'RVT_regs')] = None
        physio_reg_dict[(sbj,ses,'card_regs')] = None
        physio_reg_dict[(sbj,ses,'resp_regs')] = None


# In[72]:


with open(physio_regressors_sf_path, 'wb') as f:
    pickle.dump(physio_reg_dict, f)


# ### Output Schema: `{DATASET}_Physiological_Regressors.pkl`
# 
# - **Path**: `./summary_files/{DATASET}_Physiological_Regressors.pkl`
# - **Serialized object type**: `dict`
# - **Dictionary key (3-tuple)**:
#   - `(Subject, Session, regressor_family)` where `regressor_family` is `'RVT_regs'`, `'card_regs'`, or `'resp_regs'`
# - **Dictionary value**:
#   - Evaluation scans with an available `slibase` file: `pandas.DataFrame`
#     - Index: `TimedeltaIndex` named `'Time'`
#     - `columns.name`: `'Regressors'`
#     - Columns are the selected subset of `slibase_obj.labels` retained for the GS physio model.
#   - Missing/unavailable scans and the discovery dataset: `None`
# 
# ### Downstream notebooks that load this file
# 
# - `N17_Data_Exploration_Dashboard.ipynb`
# 
