#!/usr/bin/env python
# coding: utf-8

# # Description: Gather results and create final data structures
# 
# This notebook gather the following information for all scans in a given dataset and then save them into disk in a single location. This is done to avoid having to do all this work repeteadly in many other notebooks. 
# 
# Keep in mind that if you run any notebook that regenerates any of the metrics below, you must also rerun this notebook so the summary files in the ```summary_files``` folder (the one used by other notebooks) also get updated.
# 
# Metrics this notebook compiles:
# 
# * FC matrices for all scans and metrics
# * pBOLD estimates
# * TSNR estimates
# * Tedana component statistics and info
# * Physiological traces in clean dataframe format
# 

# In[1]:


import os.path as osp
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from utils.basics import ATLASES_DIR, CODE_DIR, PRCS_DATA_DIR, DOWNLOAD_DIRS
from utils.basics import get_dataset_index, echo_pairs_tuples, get_altas_info, pairs_of_echo_pairs
from tqdm.notebook import tqdm

from afnipy import lib_physio_reading as lpr
from afnipy import lib_physio_opts    as lpo
import copy


# Select target dataset

# In[2]:


DATASET = input ('Select Dataset (discovery or evaluation):')
DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]


# Get a list of pre-processing pipelines

# In[3]:


CENSORING_MODE='ALL'
pp_opts = {'No Censoring | No Regression':f'{CENSORING_MODE}_NoRegression',
           'No Censoring | Basic':f'{CENSORING_MODE}_Basic',
           'No Censoring | GSR':f'{CENSORING_MODE}_GS',
           'No Censoring | Tedana (fastica)':f'{CENSORING_MODE}_Tedana-fastica'}


# Set parcellation information location

# In[4]:


ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)


# In[5]:


roi_info_df, _ = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# Get list of scans in the dataset

# In[6]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# ***
# # 1. Gather Functional Connectivity Estimates

# In[7]:


fc_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_FC.pkl')
print('++ FC will be saved in %s' % fc_path)


# In[8]:


get_ipython().run_cell_magic('time', '', 'data_fc = {}\nprint(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for nordic in [\'off\',\'on\']:\n        for pp in pp_opts.values():\n            for (e_x,e_y) in echo_pairs_tuples:\n                d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                # Compose path to input TS\n                roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                # Load TS into memory\n                if (not osp.exists(roi_ts_path_x)) or (not osp.exists(roi_ts_path_y)):\n                    print(f\'++ WARNING: Missing input files for {sbj},{ses},{e_x},{e_y},{nordic},{pp}\')\n                    print(f\'            {roi_ts_path_x}\')\n                    print(f\'            {roi_ts_path_y}\')\n                    i+=1\n                    continue\n                roi_ts_x      = np.loadtxt(roi_ts_path_x)\n                roi_ts_y      = np.loadtxt(roi_ts_path_y)\n                aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df[\'ROI_Name\'].values)\n                aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df[\'ROI_Name\'].values)\n                # Compute the full correlation matrix between aux_ts_x and aux_ts_y\n                aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'R\']  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)\n                data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'C\']  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(fc_path, \'wb\') as f:\n    pickle.dump(data_fc, f)\nprint(\'[DONE]\')\n')


# ***
# 
# # 2. Gather pBOLD estimates

# In[9]:


pBOLD_xr_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_pBOLD.nc')
print('++ pBOLD will be saved in %s' % pBOLD_xr_path)


# In[10]:


pBOLD_xr = xr.DataArray(dims=['sbj','ses','pp','nordic','fc_metric','ee_vs_ee','qc_metric',],
                         coords={'sbj':       sbj_list,
                                 'ses':       ses_list,
                                 'pp':        list(pp_opts.values()),
                                 'nordic':    ['off','on'],
                                 'fc_metric': ['corr','cov'],
                                 'ee_vs_ee':  pairs_of_echo_pairs+['scan'],
                                 'qc_metric': ['pBOLD']})


# In[11]:


get_ipython().run_cell_magic('time', '', 'print(f"++ INFO: Loading all pBOLD estimates for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for NORDIC in [\'off\',\'on\']:\n            for pp in pp_opts.values():\n                for fc_metric in [\'corr\',\'cov\']:\n                    pBOLD_path    = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-{NORDIC}\',f\'errts.{sbj}.r01.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.pBOLD_{fc_metric}.csv\')\n                    try:\n                        pBOLD = pd.read_csv(pBOLD_path)\n                    except:\n                        print(\'Problematic file: %s\' % pBOLD_path )\n                    pBOLD_xr.loc[sbj,ses,pp,NORDIC,fc_metric,pBOLD[\'ee_vs_ee\'].to_list()] = pBOLD[\'pBOLD\'].values.reshape(len(pairs_of_echo_pairs)+1,1)\n')


# In[12]:


get_ipython().run_cell_magic('time', '', "print('++ INFO: Saving to disk...',end='')\npBOLD_xr.to_netcdf(pBOLD_xr_path)\nprint('[DONE]')\n")


# ***
# # 3. Gather TSNR estimates

# In[13]:


TSNR_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_TSNR.pkl')
print('++ TSNR will be saved in %s' % TSNR_path)


# In[14]:


get_ipython().run_cell_magic('time', '', 'TSNR = {}\nprint(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor TSNR_metric in [\'TSNR (Full Brain)\',\'TSNR (Visual Cortex)\']:\n    aux_df = pd.DataFrame(columns=[\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\',TSNR_metric])\n    aux_df.set_index([\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\'], inplace=True)\n    for sbj,ses in tqdm(list(ds_index), desc=TSNR_metric):\n        partial_key = (sbj, ses)\n        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)\n        if not sbj_ses_in_fc:\n            print(\'++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.\' % (sbj,ses))\n            continue\n        for pp in pp_opts.values():\n            for nordic in [\'off\',\'on\']:            \n                d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                if TSNR_metric == \'TSNR (Visual Cortex)\':\n                    aux_rois_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,\'tsnr_stats_regress\',f\'TSNR_ROIs_e02_{pp}.txt\')\n                    aux_rois      = pd.read_csv(aux_rois_path,skiprows=3, sep=r\'\\s+\').drop(0).set_index(\'ROI_name\')\n                    aux_df.loc[sbj,ses,pp,nordic] = float(aux_rois.loc[\'GHCP-R_Primary_Visual_Cortex\',\'Tmed\'])\n                if TSNR_metric == \'TSNR (Full Brain)\':\n                    aux_fb_path   = osp.join(PRCS_DATA_DIR,sbj,d_folder,\'tsnr_stats_regress\',f\'TSNR_FB_e02_{pp}.txt\')\n                    aux_fb        = pd.read_csv(aux_fb_path,skiprows=3, sep=r\'\\s+\').drop(0).set_index(\'ROI_name\')\n                    aux_df.loc[sbj,ses,pp,nordic] = float(aux_fb.loc[\'NONE\',\'Tmed\'])\n    TSNR[\'cov\',TSNR_metric] = aux_df.reset_index()\n    TSNR[\'corr\',TSNR_metric] = aux_df.reset_index()\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(TSNR_path, \'wb\') as f:\n    pickle.dump(TSNR, f)\nprint(\'[DONE]\')\n')


# ***
# # 4. Gather Tedana ICA statistics

# In[15]:


Tedana_QC_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_Tedana_QC.pkl')
print('++ Tedana statistics will be saved in %s' % Tedana_QC_path)


# In[16]:


get_ipython().run_cell_magic('time', '', 'Tedana_QC = {}\nprint(f"++ INFO: Loading all Tedana_QC for the {DATASET} into memory...")\nfor tedana_metric in [\'#ICs (All)\',\'#ICs (Likely BOLD)\',\'#ICs (Unlikely BOLD)\',\'Var. Exp. (Likely BOLD)\',\'Var. Exp. (Unlikely BOLD)\']:\n    aux_df = pd.DataFrame(columns=[\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\',tedana_metric])\n    aux_df.set_index([\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\'], inplace=True)\n    Tedana_QC[\'C\',tedana_metric] = aux_df\n')


# In[17]:


get_ipython().run_cell_magic('time', '', 'print(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for nordic in [\'off\',\'on\']:\n        for pp in pp_opts.values():\n            if \'Tedana\' not in pp:\n                Tedana_QC[\'C\',\'#ICs (All)\'].loc[sbj,ses,pp,nordic]                = np.nan\n                Tedana_QC[\'C\',\'#ICs (Likely BOLD)\'].loc[sbj,ses,pp,nordic]        = np.nan\n                Tedana_QC[\'C\',\'#ICs (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic]      = np.nan\n                Tedana_QC[\'C\',\'Var. Exp. (Likely BOLD)\'].loc[sbj,ses,pp,nordic]   = np.nan\n                Tedana_QC[\'C\',\'Var. Exp. (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic] = np.nan\n            else:\n                tedana_type = pp.split(f\'{CENSORING_MODE}_Tedana-\')[1]\n                d_folder    = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                ica_metrics_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'tedana_{tedana_type}\',\'ica_metrics.tsv\')\n                ica_metrics              = pd.read_csv(ica_metrics_path, sep=\'\\t\').set_index(\'Component\')\n                likely_bold_components   = list(ica_metrics[ica_metrics[\'classification_tags\']==\'Likely BOLD\'].index)\n                unlikely_bold_components = list(ica_metrics[ica_metrics[\'classification_tags\']==\'Unlikely BOLD\'].index)\n                \n                Tedana_QC[\'C\',\'#ICs (All)\'].loc[sbj,ses,pp,nordic]              = ica_metrics.shape[0]\n                Tedana_QC[\'C\',\'#ICs (Likely BOLD)\'].loc[sbj,ses,pp,nordic]      = len(likely_bold_components)\n                Tedana_QC[\'C\',\'#ICs (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic]    = len(unlikely_bold_components)\n                Tedana_QC[\'C\',\'Var. Exp. (Likely BOLD)\'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[likely_bold_components,\'variance explained\'].sum().round(2)\n                Tedana_QC[\'C\',\'Var. Exp. (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[unlikely_bold_components,\'variance explained\'].sum().round(2)\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(Tedana_QC_path, \'wb\') as f:\n    pickle.dump(Tedana_QC, f)\nprint(\'[DONE]\')\n')


# ***
# 
# # 5. Gather Physiological Recordings

# In[18]:


filename = f'./summary_files/{DATASET}_Physiological_Timeseries.pkl'
print(filename)


# In[19]:


get_ipython().run_cell_magic('time', '', '\nphysio_dict = {}\nfor sbj,ses in tqdm(ds_index):\n    if DATASET == \'evaluation\':\n        slibase_file = osp.join(PRCS_DATA_DIR,sbj,\'D06_Physio\',f\'{sbj}_{ses}_task-rest_echo-1_slibase.1D\')\n        if osp.exists(slibase_file):\n            # Load Physio by creating Afni RetroObj\n            phys_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.tsv.gz\')\n            json_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.json\')\n            dset_epi  = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_echo-1_bold.nii.gz\')\n            input_line = [\'./physio_calc.py\', \'-phys_file\', phys_file, \'-phys_json\', json_file, \'-dset_epi\', dset_epi, \n                            \'-prefilt_mode\', \'median\', \'-prefilt_max_freq\', \'50\', \'-verb\',\'0\']\n            args_orig  = copy.deepcopy(input_line)\n            args_dict  = lpo.main_option_processing( input_line )\n            retobj     = lpr.retro_obj( args_dict, args_orig=args_orig )\n            physio_start_time = retobj.start_time\n            # Extract Cardiac Timseries\n            card_end_time  = retobj.data[\'card\'].end_time\n            card_nsamples  = retobj.data[\'card\'].ts_orig.shape[0]\n            card_samp_rate = retobj.data[\'card\'].samp_rate\n            card_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=card_nsamples, freq=str(card_samp_rate)+"s")\n            card_df        = pd.DataFrame(retobj.data[\'card\'].ts_orig, index=card_index,columns=[\'PGG\'])\n            card_df.index.name = \'Time\'\n\n            # Extract Respiratory Timeseries\n            resp_end_time  = retobj.data[\'resp\'].end_time\n            resp_nsamples  = retobj.data[\'resp\'].ts_orig.shape[0]\n            resp_samp_rate = retobj.data[\'resp\'].samp_rate\n            resp_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=resp_nsamples, freq=str(resp_samp_rate)+"s")\n            resp_df        = pd.DataFrame(retobj.data[\'resp\'].ts_orig, index=resp_index,columns=[\'Respiration\'])\n            resp_df.index.name = \'Time\'\n\n            #Add to final dictionary\n            physio_dict[(sbj,ses,\'card\')] = card_df\n            physio_dict[(sbj,ses,\'resp\')] = resp_df\n        else:\n            physio_dict[(sbj,ses,\'card\')] = None\n            physio_dict[(sbj,ses,\'resp\')] = None\n    else:\n        physio_dict[(sbj,ses,\'card\')] = None\n        physio_dict[(sbj,ses,\'resp\')] = None\n')

