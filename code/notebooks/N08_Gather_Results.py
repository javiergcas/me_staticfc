#!/usr/bin/env python
# coding: utf-8

# # Description: Gather results and create final data structures
# 
# * FC matrices for all scans and metrics
# *

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


# In[2]:


DATASET = input ('Select Dataset (discovery or evaluation):')
DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]


# In[3]:


CENSORING_MODE='ALL'
pp_opts = {'No Censoring | No Regression':f'{CENSORING_MODE}_NoRegression',
           'No Censoring | Basic':f'{CENSORING_MODE}_Basic',
           'No Censoring | GSR':f'{CENSORING_MODE}_GS',
           'No Censoring | Tedana (fastica)':f'{CENSORING_MODE}_Tedana-fastica'}


# In[4]:


ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)


# In[5]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# In[6]:


roi_info_df, _ = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# ***
# # 1. FC matrices

# In[7]:


fc_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_FC.pkl')
print('++ FC will be saved in %s' % fc_path)


# In[8]:


get_ipython().run_cell_magic('time', '', 'data_fc = {}\nprint(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for nordic in [\'off\',\'on\']:\n        for pp in pp_opts.values():\n            for (e_x,e_y) in echo_pairs_tuples:\n                d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                # Compose path to input TS\n                roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                # Load TS into memory\n                if (not osp.exists(roi_ts_path_x)) or (not osp.exists(roi_ts_path_y)):\n                    print(f\'++ WARNING: Missing input files for {sbj},{ses},{e_x},{e_y},{nordic},{pp}\')\n                    print(f\'            {roi_ts_path_x}\')\n                    print(f\'            {roi_ts_path_y}\')\n                    i+=1\n                    continue\n                roi_ts_x      = np.loadtxt(roi_ts_path_x)\n                roi_ts_y      = np.loadtxt(roi_ts_path_y)\n                aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df[\'ROI_Name\'].values)\n                aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df[\'ROI_Name\'].values)\n                # Compute the full correlation matrix between aux_ts_x and aux_ts_y\n                aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'R\']  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)\n                data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'C\']  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(fc_path, \'wb\') as f:\n    pickle.dump(data_fc, f)\nprint(\'[DONE]\')\n')


# ***
# 
# # pBOLD statistic

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
# # 3. TSNR

# In[13]:


TSNR_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_TSNR.pkl')
print('++ TSNR will be saved in %s' % TSNR_path)


# In[14]:


get_ipython().run_cell_magic('time', '', 'TSNR = {}\nprint(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor TSNR_metric in [\'TSNR (Full Brain)\',\'TSNR (Visual Cortex)\']:\n    aux_df = pd.DataFrame(columns=[\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\',TSNR_metric])\n    aux_df.set_index([\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\'], inplace=True)\n    for sbj,ses in tqdm(list(ds_index), desc=TSNR_metric):\n        partial_key = (sbj, ses)\n        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)\n        if not sbj_ses_in_fc:\n            print(\'++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.\' % (sbj,ses))\n            continue\n        for pp in pp_opts.values():\n            for nordic in [\'off\',\'on\']:            \n                d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                if TSNR_metric == \'TSNR (Visual Cortex)\':\n                    aux_rois_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,\'tsnr_stats_regress\',f\'TSNR_ROIs_e02_{pp}.txt\')\n                    aux_rois      = pd.read_csv(aux_rois_path,skiprows=3, sep=r\'\\s+\').drop(0).set_index(\'ROI_name\')\n                    aux_df.loc[sbj,ses,pp,nordic] = float(aux_rois.loc[\'GHCP-R_Primary_Visual_Cortex\',\'Tmed\'])\n                if TSNR_metric == \'TSNR (Full Brain)\':\n                    aux_fb_path   = osp.join(PRCS_DATA_DIR,sbj,d_folder,\'tsnr_stats_regress\',f\'TSNR_FB_e02_{pp}.txt\')\n                    aux_fb        = pd.read_csv(aux_fb_path,skiprows=3, sep=r\'\\s+\').drop(0).set_index(\'ROI_name\')\n                    aux_df.loc[sbj,ses,pp,nordic] = float(aux_fb.loc[\'NONE\',\'Tmed\'])\n    TSNR[\'cov\',TSNR_metric] = aux_df.reset_index()\n    TSNR[\'corr\',TSNR_metric] = aux_df.reset_index()\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(TSNR_path, \'wb\') as f:\n    pickle.dump(TSNR, f)\nprint(\'[DONE]\')\n')


# ***
# # Tedana QC matrics

# In[15]:


Tedana_QC_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_Tedana_QC.pkl')
print('++ Tedana_QC will be saved in %s' % Tedana_QC_path)


# In[16]:


get_ipython().run_cell_magic('time', '', 'Tedana_QC = {}\nprint(f"++ INFO: Loading all Tedana_QC for the {DATASET} into memory...")\nfor tedana_metric in [\'#ICs (All)\',\'#ICs (Likely BOLD)\',\'#ICs (Unlikely BOLD)\',\'Var. Exp. (Likely BOLD)\',\'Var. Exp. (Unlikely BOLD)\']:\n    aux_df = pd.DataFrame(columns=[\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\',tedana_metric])\n    aux_df.set_index([\'Subject\',\'Session\',\'Pre-processing\',\'NORDIC\'], inplace=True)\n    Tedana_QC[\'C\',tedana_metric] = aux_df\n')


# In[17]:


get_ipython().run_cell_magic('time', '', 'print(f"++ INFO: Loading all FC matrices for the {DATASET} into memory...")\nfor sbj,ses in tqdm(list(ds_index)):\n    for nordic in [\'off\',\'on\']:\n        for pp in pp_opts.values():\n            if \'Tedana\' not in pp:\n                Tedana_QC[\'C\',\'#ICs (All)\'].loc[sbj,ses,pp,nordic]                = np.nan\n                Tedana_QC[\'C\',\'#ICs (Likely BOLD)\'].loc[sbj,ses,pp,nordic]        = np.nan\n                Tedana_QC[\'C\',\'#ICs (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic]      = np.nan\n                Tedana_QC[\'C\',\'Var. Exp. (Likely BOLD)\'].loc[sbj,ses,pp,nordic]   = np.nan\n                Tedana_QC[\'C\',\'Var. Exp. (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic] = np.nan\n            else:\n                tedana_type = pp.split(f\'{CENSORING_MODE}_Tedana-\')[1]\n                d_folder    = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                ica_metrics_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'tedana_{tedana_type}\',\'ica_metrics.tsv\')\n                ica_metrics              = pd.read_csv(ica_metrics_path, sep=\'\\t\').set_index(\'Component\')\n                likely_bold_components   = list(ica_metrics[ica_metrics[\'classification_tags\']==\'Likely BOLD\'].index)\n                unlikely_bold_components = list(ica_metrics[ica_metrics[\'classification_tags\']==\'Unlikely BOLD\'].index)\n                \n                Tedana_QC[\'C\',\'#ICs (All)\'].loc[sbj,ses,pp,nordic]              = ica_metrics.shape[0]\n                Tedana_QC[\'C\',\'#ICs (Likely BOLD)\'].loc[sbj,ses,pp,nordic]      = len(likely_bold_components)\n                Tedana_QC[\'C\',\'#ICs (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic]    = len(unlikely_bold_components)\n                Tedana_QC[\'C\',\'Var. Exp. (Likely BOLD)\'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[likely_bold_components,\'variance explained\'].sum().round(2)\n                Tedana_QC[\'C\',\'Var. Exp. (Unlikely BOLD)\'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[unlikely_bold_components,\'variance explained\'].sum().round(2)\nprint(\'++ INFO: Saving to disk...\',end=\'\')\nwith open(Tedana_QC_path, \'wb\') as f:\n    pickle.dump(Tedana_QC, f)\nprint(\'[DONE]\')\n')


# ***
# 
# # Physiological Traces

# In[18]:


filename = f'./summary_files/{DATASET}_Physiological_Timeseries.pkl'
print(filename)


# In[19]:


get_ipython().run_cell_magic('time', '', '\nphysio_dict = {}\nfor sbj,ses in tqdm(ds_index):\n    if DATASET == \'evaluation\':\n        slibase_file = osp.join(PRCS_DATA_DIR,sbj,\'D06_Physio\',f\'{sbj}_{ses}_task-rest_echo-1_slibase.1D\')\n        if osp.exists(slibase_file):\n            # Load Physio by creating Afni RetroObj\n            phys_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.tsv.gz\')\n            json_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.json\')\n            dset_epi  = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_echo-1_bold.nii.gz\')\n            input_line = [\'./physio_calc.py\', \'-phys_file\', phys_file, \'-phys_json\', json_file, \'-dset_epi\', dset_epi, \n                            \'-prefilt_mode\', \'median\', \'-prefilt_max_freq\', \'50\', \'-verb\',\'0\']\n            args_orig  = copy.deepcopy(input_line)\n            args_dict  = lpo.main_option_processing( input_line )\n            retobj     = lpr.retro_obj( args_dict, args_orig=args_orig )\n            physio_start_time = retobj.start_time\n            # Extract Cardiac Timseries\n            card_end_time  = retobj.data[\'card\'].end_time\n            card_nsamples  = retobj.data[\'card\'].ts_orig.shape[0]\n            card_samp_rate = retobj.data[\'card\'].samp_rate\n            card_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=card_nsamples, freq=str(card_samp_rate)+"s")\n            card_df        = pd.DataFrame(retobj.data[\'card\'].ts_orig, index=card_index,columns=[\'PGG\'])\n            card_df.index.name = \'Time\'\n\n            # Extract Respiratory Timeseries\n            resp_end_time  = retobj.data[\'resp\'].end_time\n            resp_nsamples  = retobj.data[\'resp\'].ts_orig.shape[0]\n            resp_samp_rate = retobj.data[\'resp\'].samp_rate\n            resp_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=resp_nsamples, freq=str(resp_samp_rate)+"s")\n            resp_df        = pd.DataFrame(retobj.data[\'resp\'].ts_orig, index=resp_index,columns=[\'Respiration\'])\n            resp_df.index.name = \'Time\'\n\n            #Add to final dictionary\n            physio_dict[(sbj,ses,\'card\')] = card_df\n            physio_dict[(sbj,ses,\'resp\')] = resp_df\n        else:\n            physio_dict[(sbj,ses,\'card\')] = None\n            physio_dict[(sbj,ses,\'resp\')] = None\n    else:\n        physio_dict[(sbj,ses,\'card\')] = None\n        physio_dict[(sbj,ses,\'resp\')] = None\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


with open(filename, 'wb') as f:
       pickle.dump(physio_dict, f)


# In[ ]:





# In[27]:


Physio_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_Physio.pkl')
print('++ Physio will be saved in %s' % Physio_path)


# In[30]:


Physio={}
print(f"++ INFO: Loading Physio information for the {DATASET} into memory...")
if DATASET == 'evaluation':
    report_card_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_card_review_all_scans.txt')
    report_card_summary_df    = read_group_physio_reports(report_card_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_card_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_card = report_card_summary_df['peak ival over dset mean std'].copy()
    df_card.columns=['Mean','St.Dev.']
    df_card['color'] = ['red' if c else 'green' for c in outliers]
    
    Physio[('corr','Physio (cardiac)')] = df_card
    Physio[('cov','Physio (cardiac)')] = df_card
else:
    Physio[('corr','Physio (cardiac)')] = None
    Physio[('cov','Physio (cardiac)')] = None

print('++ INFO: Saving to disk...',end='')
with open(Physio_path, 'wb') as f:
    pickle.dump(Physio, f)
print('[DONE]')


# ***
# # Global Signal

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)


# In[2]:


import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import pickle
import panel as pn
from nilearn.connectome import sym_matrix_to_vec


# In[3]:


from utils.basics import compute_residuals, echo_pairs, pairs_of_echo_pairs, echo_pairs_tuples, get_dataset_index, get_altas_info
from utils.basics import TES_MSEC
from utils.basics import ATLASES_DIR, PRCS_DATA_DIR, PRJ_DIR, FMRI_FINAL_NUM_SAMPLES, FMRI_TRS, NUM_DISCARDED_VOLUMES, DOWNLOAD_DIRS
from utils.basics import mse_dist, chord_distance_between_intersecting_lines, read_group_physio_reports
from utils.dashboard import fc_across_echoes_scatter_page, get_fc_matrices, get_static_report, get_fc_matrix,dynamic_summary_plot_gated, get_ts_report_page

from sklearn.ensemble import IsolationForest


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ***
# 
# # Select the dataset you want to work this next:

# In[4]:


DATASET = input ('Select Dataset (discovery or evaluation):')


# In[5]:


CENSORING_MODE='ALL'


# In[6]:


fMRI_Sampling_Rate      = FMRI_TRS[DATASET]
fMRI_Preproc_Nsamples   = FMRI_FINAL_NUM_SAMPLES[DATASET]
fMRI_Ndiscard_samples   = NUM_DISCARDED_VOLUMES[DATASET]
fMRI_Preproc_Start_Time = str(float(fMRI_Sampling_Rate.replace('s','')) * int(fMRI_Ndiscard_samples))+'s'
print('++ fMRI Sampling Rate            = %s' % fMRI_Sampling_Rate)
print('++ fMRI Final Number of Samples  = %d Acquisitions' % fMRI_Preproc_Nsamples)
print('++ fMRI Number discarded Samples = %d Acquisitions' % fMRI_Ndiscard_samples)
print('++ fMRI Preproc Start Time       = %s' % fMRI_Preproc_Start_Time)


# In[7]:


DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]


# In[8]:


echo_times_dict = TES_MSEC[DATASET]
print(echo_times_dict)


# In[9]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# In[10]:


fMRI_Preproc_index = pd.timedelta_range(start=fMRI_Preproc_Start_Time, periods=fMRI_Preproc_Nsamples, freq=fMRI_Sampling_Rate)
fMRI_Preproc_index[0:3]


# # 1. Basic Information
# Create lists with all 6 possible echo combinations, and then all possible pairings between those.

# In[11]:


print('Echo Pairs[n=%d] = %s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d] = %s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))


# ***
# # 2. Load Atlas Information

# In[12]:


ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)


# In[13]:


roi_info_df, power264_nw_cmap = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# ***
# # 3. Load Timeseries and compute R and C matrices
# 
# This cell will load ROI Timeseries, compute R and C, and place these into a dictionary of datafrmes. It will do this for the Basic denoising pipeline (Basic) and no censoring (ALL).

# In[15]:


if CENSORING_MODE == 'ALL':
    pp_opts = {'No Censoring | No Regression':f'{CENSORING_MODE}_NoRegression',
           'No Censoring | Basic':f'{CENSORING_MODE}_Basic',
           'No Censoring | GSR':f'{CENSORING_MODE}_GS',
           'No Censoring | Tedana (fastica - paper)':f'{CENSORING_MODE}_Tedana-fastica',
           'No Censoring | Tedana (fastica - new)':f'{CENSORING_MODE}_tedana-fastica'}
else:
    pp_opts = {'Censoring | No Regression':f'{CENSORING_MODE}_NoRegression',
               'Censoring | Basic':f'{CENSORING_MODE}_Basic',
           'Censoring | GSR':f'{CENSORING_MODE}_GS',
           'Censoring | Tedana (fastica - paper)':f'{CENSORING_MODE}_Tedana-fastica',
           'Censoring | Tedana (fastica - new)':f'{CENSORING_MODE}_tedana-fastica'}
nordic_opts = {'Do not use':'off', 'Active':'on'}

data_fc = {}


# In[16]:


get_ipython().run_cell_magic('time', '', 'filename = f\'./cache/{DATASET}_fc_{CENSORING_MODE}.pkl\'\nprint(filename)\ni=0\nif osp.exists(filename):\n    print("++ WARNING: Loading pre-existing data from cache folder.")\n    with open(filename, \'rb\') as f:\n        data_fc = pickle.load(f)\nelse:\n    for sbj,ses in tqdm(list(ds_index)):\n        for nordic in nordic_opts.values():\n            for pp in pp_opts.values():\n                for (e_x,e_y) in echo_pairs_tuples:\n                    d_folder = f\'D03_Preproc_{ses}_NORDIC-{nordic}\'\n                    # Compose path to input TS\n                    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f\'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts\')\n                    # Load TS into memory\n                    if (not osp.exists(roi_ts_path_x)) or (not osp.exists(roi_ts_path_y)):\n                        print(f\'++ WARNING: Missing input files for {sbj},{ses},{e_x},{e_y},{nordic},{pp}\')\n                        print(f\'            {roi_ts_path_x}\')\n                        print(f\'            {roi_ts_path_y}\')\n                        i+=1\n                        continue\n                    roi_ts_x      = np.loadtxt(roi_ts_path_x)\n                    roi_ts_y      = np.loadtxt(roi_ts_path_y)\n                    aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df[\'ROI_Name\'].values)\n                    aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df[\'ROI_Name\'].values)\n                    # Compute the full correlation matrix between aux_ts_x and aux_ts_y\n                    aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                    aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]\n                    data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'R\']  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)\n                    data_fc[sbj, ses, pp,nordic,\'|\'.join((e_x,e_y)),\'C\']  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)\n    with open(filename, \'wb\') as f:\n        pickle.dump(data_fc, f)\nprint(i)\n')


# ***
# 
# # 4. Compute pBOLD
# 
# ## 4.1. First compute pBOLD on each separate scatter plot

# In[20]:


get_ipython().run_cell_magic('time', '', 'filename = f\'./cache/{DATASET}_pBOLD_all_scatters_{CENSORING_MODE}.nc\'\nif osp.exists(filename):\n    print("++ WARNING: Loading pre-existing data from cache folder.")\n    pBOLD_xr = xr.open_dataarray(filename)\nelse:\n    pBOLD_xr = xr.DataArray(dims=[\'sbj\',\'ses\',\'pp\',\'nordic\',\'fc_metric\',\'ee_vs_ee\',\'qc_metric\',],\n                         coords={\'sbj\':       sbj_list,\n                                 \'ses\':       ses_list,\n                                 \'pp\':        list(pp_opts.values()),\n                                 \'nordic\':    list(nordic_opts.values()),\n                                 \'fc_metric\': [\'R\',\'C\'],\n                                 \'ee_vs_ee\':  pairs_of_echo_pairs,\n                                 \'qc_metric\': [\'pBOLD\',\'pSo\']})\n    for sbj in tqdm(sbj_list):\n        for ses in ses_list:\n            partial_key = (sbj, ses)\n            sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)\n            if not sbj_ses_in_fc:\n                print(\'++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.\' % (sbj,ses))\n                continue\n            for fc_metric in [\'C\',\'R\']:\n                for pp in pp_opts.values():\n                    for nordic in nordic_opts.values():\n                        for eep in pairs_of_echo_pairs:\n                            # Extract vectorized FC for this particular case\n                            # ==============================================\n                            eep1,eep2 = eep.split(\'_vs_\')\n                            data_df = pd.DataFrame(columns=[eep1,eep2])\n                            data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)\n                            data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)\n                            # Calculate slope and intercept for the two extreme scenarios\n                            # ===========================================================\n                            So_line_sl, So_line_int = 1.,0. # This is always the same\n                            BOLD_line_int = 0.              # This is always the same\n                            if fc_metric  == \'R\':\n                               BOLD_line_sl = 1.\n                            if fc_metric == \'C\':\n                                e1_X,e2_X     = eep1.split(\'|\')\n                                e1_Y,e2_Y     = eep2.split(\'|\')\n                                BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])\n                            # QC1. Compute dBOLD and dSo metrics\n                            # ==================================\n                            pBOLD_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,\'pBOLD\'],pBOLD_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,\'pSo\'] = mse_dist(data_df.values,\n                                                                                                                                           BOLD_line_sl,\n                                                                                                                                           So_line_sl, \n                                                                                                                                           weight_fn=lambda r: np.power(r,1.0), \n                                                                                                                                           max_weight_fn=lambda r: np.minimum(r,np.quantile(r,.95)),\n                                                                                                                                           tol=1e-3)\n    pBOLD_xr.to_netcdf(filename)\n')


# In[21]:


pBOLD_xr.sel(sbj='MGSBJ01',ses='constant_gated',fc_metric='C', nordic='off',qc_metric='pBOLD',pp='ALL_NoRegression')


# ## 4.2. Weigthed average across scatter plots
# 
# Becuase the separation of the BOLD and non-BOLD line is dependent on the contributing echoes, instead of simply averaging all scatter-specific pBOLD values, we propose to do a weigthed average where the weights correspond to the chord distance between the lines at radius = 1.0.
# 
# The next cell computes this distance for all available scatter plots.

# In[19]:


scat_plot_weights = xr.DataArray(np.zeros(len(pairs_of_echo_pairs)),dims='ee_vs_ee',coords={'ee_vs_ee':pairs_of_echo_pairs})
for ppe in pairs_of_echo_pairs:
    ep1,ep2 = ppe.split('_vs_')
    ex1,ex2 = ep1.split('|')
    ey1,ey2 = ep2.split('|')
    this_case_BOLD_slope = (echo_times_dict[ey1] * echo_times_dict[ey2]) / (echo_times_dict[ex1] * echo_times_dict[ex2])
    scat_plot_weights.loc[ppe] = chord_distance_between_intersecting_lines(1.0, this_case_BOLD_slope, r=0.5)


# In[20]:


scat_plot_weights.to_dataframe(name='chord').sort_values(by='chord',ascending=False)


# Here, we know calculate the final metrics per dataset. Only pBOLD will use the weights during the average. TSNR will be simply averaged.

# In[21]:


QC_metrics = {}
for fc_metric in ['R','C']:
    for qc_metric in ['pBOLD','pSo']:
        aux_df = pBOLD_xr.weighted(scat_plot_weights).mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
        aux_df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
        QC_metrics[(fc_metric,qc_metric)] = aux_df
QC_metrics['C','pBOLD'].head(5)


# # 5. Gather TSNR information

# In[22]:


get_ipython().run_cell_magic('time', '', "for TSNR_metric in ['TSNR (Full Brain)','TSNR (Visual Cortex)']:\n    aux_df = pd.DataFrame(columns=['Subject','Session','Pre-processing','NORDIC',TSNR_metric])\n    aux_df.set_index(['Subject','Session','Pre-processing','NORDIC'], inplace=True)\n    for sbj in tqdm(sbj_list, desc=TSNR_metric):\n        for ses in ses_list:\n            partial_key = (sbj, ses)\n            sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)\n            if not sbj_ses_in_fc:\n                print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))\n                continue\n            for pp in pp_opts.values():\n                for nordic in nordic_opts.values():            \n                    d_folder = f'D03_Preproc_{ses}_NORDIC-{nordic}'\n                    if TSNR_metric == 'TSNR (Visual Cortex)':\n                        aux_rois_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tsnr_stats_regress',f'TSNR_ROIs_e02_{pp}.txt')\n                        aux_rois      = pd.read_csv(aux_rois_path,skiprows=3, sep=r'\\s+').drop(0).set_index('ROI_name')\n                        aux_df.loc[sbj,ses,pp,nordic] = float(aux_rois.loc['GHCP-R_Primary_Visual_Cortex','Tmed'])\n                    if TSNR_metric == 'TSNR (Full Brain)':\n                        aux_fb_path   = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tsnr_stats_regress',f'TSNR_FB_e02_{pp}.txt')\n                        aux_fb        = pd.read_csv(aux_fb_path,skiprows=3, sep=r'\\s+').drop(0).set_index('ROI_name')\n                        aux_df.loc[sbj,ses,pp,nordic] = float(aux_fb.loc['NONE','Tmed'])\n    QC_metrics['C',TSNR_metric] = aux_df.reset_index()\n    QC_metrics['R',TSNR_metric] = aux_df.reset_index()\n")


# ***
# # 6. Tedana derived metrics

# In[23]:


get_ipython().run_cell_magic('time', '', "for tedana_metric in ['#ICs (All)','#ICs (Likely BOLD)','#ICs (Unlikely BOLD)','Var. Exp. (Likely BOLD)','Var. Exp. (Unlikely BOLD)']:\n    aux_df = pd.DataFrame(columns=['Subject','Session','Pre-processing','NORDIC',tedana_metric])\n    aux_df.set_index(['Subject','Session','Pre-processing','NORDIC'], inplace=True)\n    QC_metrics['C',tedana_metric] = aux_df\n    \nfor sbj in tqdm(sbj_list):\n    for ses in ses_list:\n        partial_key = (sbj, ses)\n        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)\n        if not sbj_ses_in_fc:\n            print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))\n            continue\n        for nordic in nordic_opts.values():\n            for pp in pp_opts.values():\n                if 'Tedana' not in pp:\n                    QC_metrics['C','#ICs (All)'].loc[sbj,ses,pp,nordic]                = np.nan\n                    QC_metrics['C','#ICs (Likely BOLD)'].loc[sbj,ses,pp,nordic]        = np.nan\n                    QC_metrics['C','#ICs (Unlikely BOLD)'].loc[sbj,ses,pp,nordic]      = np.nan\n                    QC_metrics['C','Var. Exp. (Likely BOLD)'].loc[sbj,ses,pp,nordic]   = np.nan\n                    QC_metrics['C','Var. Exp. (Unlikely BOLD)'].loc[sbj,ses,pp,nordic] = np.nan\n                else:\n                    tedana_type = pp.split(f'{CENSORING_MODE}_Tedana-')[1]\n                    d_folder    = f'D03_Preproc_{ses}_NORDIC-{nordic}'\n                    ica_metrics_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'tedana_{tedana_type}','ica_metrics.tsv')\n                    ica_metrics              = pd.read_csv(ica_metrics_path, sep='\\t').set_index('Component')\n                    likely_bold_components   = list(ica_metrics[ica_metrics['classification_tags']=='Likely BOLD'].index)\n                    unlikely_bold_components = list(ica_metrics[ica_metrics['classification_tags']=='Unlikely BOLD'].index)\n                    \n                    QC_metrics['C','#ICs (All)'].loc[sbj,ses,pp,nordic]              = ica_metrics.shape[0]\n                    QC_metrics['C','#ICs (Likely BOLD)'].loc[sbj,ses,pp,nordic]      = len(likely_bold_components)\n                    QC_metrics['C','#ICs (Unlikely BOLD)'].loc[sbj,ses,pp,nordic]    = len(unlikely_bold_components)\n                    QC_metrics['C','Var. Exp. (Likely BOLD)'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[likely_bold_components,'variance explained'].sum().round(2)\n                    QC_metrics['C','Var. Exp. (Unlikely BOLD)'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[unlikely_bold_components,'variance explained'].sum().round(2)\n")


# ***
# 
# KEEP RUNNING HERE!!!!!!!
# # 7. Physiological Recording Derived Metrics

# In[24]:


if DATASET == 'evaluation':
    report_card_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_card_review_all_scans.txt')
    report_card_summary_df    = read_group_physio_reports(report_card_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_card_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_card = report_card_summary_df['peak ival over dset mean std'].copy()
    df_card.columns=['Mean','St.Dev.']
    df_card['color'] = ['red' if c else 'green' for c in outliers]
    
    QC_metrics[('R','Physio (cardiac)')] = df_card
    QC_metrics[('C','Physio (cardiac)')] = df_card
else:
    QC_metrics[('R','Physio (cardiac)')] = None
    QC_metrics[('C','Physio (cardiac)')] = None


# In[25]:


if DATASET == 'evaluation':
    report_resp_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_resp_review_all_scans.txt')
    report_resp_summary_df    = read_group_physio_reports(report_resp_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_resp_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_resp = report_resp_summary_df['peak ival over dset mean std'].copy()
    df_resp.columns=['Mean','St.Dev.']
    df_resp['color'] = ['red' if c else 'green' for c in outliers]
    
    QC_metrics[('R','Physio (resp)')] = df_resp
    QC_metrics[('C','Physio (resp)')] = df_resp
else:
    QC_metrics[('R','Physio (resp)')] = None
    QC_metrics[('C','Physio (resp)')] = None


# In[26]:


for tedana_metric in ['#ICs (All)','#ICs (Likely BOLD)','#ICs (Unlikely BOLD)','Var. Exp. (Likely BOLD)','Var. Exp. (Unlikely BOLD)']:
    QC_metrics['C',tedana_metric] = QC_metrics['C',tedana_metric].reset_index()
    QC_metrics['R',tedana_metric] = QC_metrics['C',tedana_metric].copy()


# In[27]:


import pickle 
with open(f'./cache/{DATASET}_QC_metrics_{CENSORING_MODE}.pkl', 'wb') as f:
    pickle.dump(QC_metrics, f)


# # 8. Load the Global Signal Timeseries

# In[30]:


if DATASET == 'evaluation':
    kappa_rho_df = pd.read_csv(f'./cache/{DATASET}_gs_kappa_rho.{CENSORING_MODE}.csv', index_col=[0,1])
    print("++ INFO: The shape of kappa_rho_df is %s" % str(kappa_rho_df.shape))
else:
    kappa_rho_df = None


# In[31]:


get_ipython().run_cell_magic('time', '', 'filename = f\'./cache/{DATASET}_{CENSORING_MODE}_GS_info_and_ts.pkl\'\nif osp.exists(filename):\n    print("++ WARNING: Loading pre-existing data from cache folder.")\n    with open(filename, \'rb\') as f:\n        gs_df_dict = pickle.load(f)\nelse:\n    gs_df_dict = {}\n    for sbj,ses in tqdm(ds_index):\n        if DATASET == \'evaluation\':\n            # Load All the GS versions (each echo time and optimally combined)\n            gs_e01_path = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',f\'pb03.{sbj}.r01.e01.volreg.GS.1D\')\n            gs_e02_path = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',f\'pb03.{sbj}.r01.e02.volreg.GS.1D\')\n            gs_e03_path = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',f\'pb03.{sbj}.r01.e03.volreg.GS.1D\')\n            gs_OC_path  = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',f\'pb06.{sbj}.r01.tedana_fastica_OC.GS.1D\')\n            gs_e01 = np.loadtxt(gs_e01_path)\n            gs_e02 = np.loadtxt(gs_e02_path)\n            gs_e03 = np.loadtxt(gs_e03_path)\n            gs_OC  = np.loadtxt(gs_OC_path)\n            gs_df = pd.DataFrame([gs_OC,gs_e01,gs_e02,gs_e03],index=[\'OC\',\'TE1\',\'TE2\',\'TE3\']).T\n            gs_df = gs_df.infer_objects()\n            gs_df.index = fMRI_Preproc_index\n            gs_df.index.name = \'Time\'\n            gs_df.columns.name=\'Echo Time\'\n            # Transform to units of Signal Percent Change\n            gs_df_spc = 100*(gs_df-gs_df.mean())/gs_df.mean()\n            gs_df_dict[(sbj,ses,\'gs_ts\')] = gs_df_spc\n    \n            # Create Dataframe with Gs properties of interest\n            GS_phys_match_file = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',f\'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl\')\n            try:\n                with open(GS_phys_match_file, \'rb\') as f:\n                    loaded_dict = pickle.load(f)\n                gs_adjr2_physio = float(loaded_dict[\'model\'].rsquared_adj)\n            except:\n                gs_adjr2_physio = None\n            gs_kappa        = float(kappa_rho_df.loc[(sbj,ses),\'kappa (GS)\'])\n            gs_rho          = float(kappa_rho_df.loc[(sbj,ses),\'rho (GS)\'])\n            gs_df_metrics   = pd.DataFrame([gs_adjr2_physio,gs_kappa,gs_rho],index=[\'Adj R2 Physio\',\'kappa\',\'rho\'],columns=[\'GS\']).T\n            gs_df_dict[sbj,ses,\'gs_metrics\'] = gs_df_metrics\n        else:\n            gs_df_dict[(sbj,ses,\'gs_ts\')] = None\n            gs_df_dict[sbj,ses,\'gs_metrics\'] = None\n    with open(filename, \'wb\') as f:\n        pickle.dump(gs_df_dict, f)\n')


# # 9. Load ICA Timeseries and basic statistics

# In[32]:


get_ipython().run_cell_magic('time', '', 'filename = f\'./cache/{DATASET}_{CENSORING_MODE}_ICAs.pkl\'\nprint(filename)\nif osp.exists(filename):\n    print("++ WARNING: Loading pre-existing data from cache folder.")\n    with open(filename, \'rb\') as f:\n        ica_dict = pickle.load(f)\nelse:\n    ica_dict = {}\n    for sbj,ses in tqdm(ds_index):\n        # Load IC Timeseries\n        ic_ts_path         = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',\'tedana_fastica\',f\'ica_mixing.tsv\')\n        ic_ts              = pd.read_csv(ic_ts_path, sep=\'\\t\')\n        ic_ts              = ic_ts.infer_objects()\n        ic_ts.index        = fMRI_Preproc_index\n        ic_ts.index.name   = \'Time\'\n        ic_ts.columns.name = \'Components\'\n        ica_dict[(sbj,ses,\'ic_ts\')] = ic_ts\n        # Load IC Properties\n        ic_metrics = pd.read_csv(osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',\'tedana_fastica\',f\'ica_metrics.tsv\'),sep=\'\\t\', index_col=0)\n        ic_metrics.index.name=\'Name\'\n        ic_metrics               = ic_metrics.round(2)[[\'kappa\',\'rho\',\'variance explained\',\'classification_tags\']]\n        if DATASET == \'evaluation\':\n            ic_metrics[\'corrwithGS\'] = ic_ts.corrwith(gs_df_dict[sbj,ses,\'gs_ts\'][\'OC\'])\n            ic_metrics.columns = [\'kappa\',\'rho\',\'varepx\',\'label\',\'R(ic,GS)\']\n        else:\n            ic_metrics[\'corrwithGS\'] = np.nan\n            ic_metrics.columns = [\'kappa\',\'rho\',\'varepx\',\'label\',\'R(ic,GS)\']\n        ica_dict[(sbj,ses,\'ic_metrics\')] = ic_metrics\n    with open(filename, \'wb\') as f:\n        pickle.dump(ica_dict, f)\n')


# # 10. Load Physiological Recordings

# In[33]:


from afnipy import lib_physio_reading as lpr
from afnipy import lib_physio_opts    as lpo
import copy


# In[34]:


get_ipython().run_cell_magic('time', '', 'filename = f\'./cache/{DATASET}_{CENSORING_MODE}_Physiological_Timeseries.pkl\'\nprint(filename)\nif osp.exists(filename):\n    print("++ WARNING: Loading pre-existing data from cache folder.")\n    with open(filename, \'rb\') as f:\n        physio_dict = pickle.load(f)\nelse:\n    physio_dict = {}\n    for sbj,ses in tqdm(ds_index):\n        if DATASET == \'evaluation\':\n            slibase_file = osp.join(PRCS_DATA_DIR,sbj,\'D06_Physio\',f\'{sbj}_{ses}_task-rest_echo-1_slibase.1D\')\n            if osp.exists(slibase_file):\n                # Load Physio by creating Afni RetroObj\n                phys_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.tsv.gz\')\n                json_file = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_physio.json\')\n                dset_epi  = osp.join(DOWNLOAD_DIR,sbj,ses,\'func\',f\'{sbj}_{ses}_task-rest_echo-1_bold.nii.gz\')\n                input_line = [\'./physio_calc.py\', \'-phys_file\', phys_file, \'-phys_json\', json_file, \'-dset_epi\', dset_epi, \n                              \'-prefilt_mode\', \'median\', \'-prefilt_max_freq\', \'50\', \'-verb\',\'0\']\n                args_orig  = copy.deepcopy(input_line)\n                args_dict  = lpo.main_option_processing( input_line )\n                retobj     = lpr.retro_obj( args_dict, args_orig=args_orig )\n                physio_start_time = retobj.start_time\n                # Extract Cardiac Timseries\n                card_end_time  = retobj.data[\'card\'].end_time\n                card_nsamples  = retobj.data[\'card\'].ts_orig.shape[0]\n                card_samp_rate = retobj.data[\'card\'].samp_rate\n                card_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=card_nsamples, freq=str(card_samp_rate)+"s")\n                card_df        = pd.DataFrame(retobj.data[\'card\'].ts_orig, index=card_index,columns=[\'PGG\'])\n                card_df.index.name = \'Time\'\n    \n                # Extract Respiratory Timeseries\n                resp_end_time  = retobj.data[\'resp\'].end_time\n                resp_nsamples  = retobj.data[\'resp\'].ts_orig.shape[0]\n                resp_samp_rate = retobj.data[\'resp\'].samp_rate\n                resp_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=resp_nsamples, freq=str(resp_samp_rate)+"s")\n                resp_df        = pd.DataFrame(retobj.data[\'resp\'].ts_orig, index=resp_index,columns=[\'Respiration\'])\n                resp_df.index.name = \'Time\'\n    \n                #Add to final dictionary\n                physio_dict[(sbj,ses,\'card\')] = card_df\n                physio_dict[(sbj,ses,\'resp\')] = resp_df\n            else:\n                physio_dict[(sbj,ses,\'card\')] = None\n                physio_dict[(sbj,ses,\'resp\')] = None\n        else:\n            physio_dict[(sbj,ses,\'card\')] = None\n            physio_dict[(sbj,ses,\'resp\')] = None\n    with open(filename, \'wb\') as f:\n        pickle.dump(physio_dict, f)\n')


# # 12. Load Physiological Regressors

# In[35]:


from afnipy.lib_afni1D import Afni1D


# In[36]:


get_ipython().run_cell_magic('time', '', 'filename = f\'./cache/{DATASET}_{CENSORING_MODE}_Physiological_Regressors.pkl\'\nprint(filename)\nif osp.exists(filename):\n    print("++ WARNING: Loading pre-existing data from cache folder.")\n    with open(filename, \'rb\') as f:\n        physio_reg_dict = pickle.load(f)\nelse:\n    physio_reg_dict = {}\n    for sbj,ses in tqdm(ds_index):\n        if DATASET == \'evaluation\':\n            slibase_path = osp.join(PRCS_DATA_DIR,sbj,\'D06_Physio\',f\'{sbj}_{ses}_task-rest_echo-1_slibase.1D\')\n            if osp.exists(slibase_path):\n                slibase_obj  = Afni1D(slibase_path)\n                slibase_df   = pd.read_csv(slibase_path, comment=\'#\', delimiter=\' +\', header=None, engine=\'python\')\n                slibase_df.columns=slibase_obj.labels\n                slibase_df=slibase_df[3::].reset_index(drop=True)\n                slibase_df.index = fMRI_Preproc_index\n                slibase_df.index.name = \'Time\'\n                slibase_df.columns.name = \'Regressors\'\n                # Load list of selected regressors for the GS\n                GS_phys_match_file = osp.join(PRCS_DATA_DIR,sbj,f\'D03_Preproc_{ses}_NORDIC-off\',f\'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl\')\n                with open(GS_phys_match_file, \'rb\') as f:\n                    loaded_dict = pickle.load(f)\n                    selected_physio_regs = loaded_dict[\'selected_regs\']\n                selected_RVT_regs  = [r for r in selected_physio_regs if \'rvt\' in r]\n                selected_card_regs = [r for r in selected_physio_regs if \'card\' in r]\n                selected_resp_regs = [r for r in selected_physio_regs if \'resp\' in r]\n                physio_reg_dict[(sbj,ses,\'RVT_regs\')]  = slibase_df[selected_RVT_regs]\n                physio_reg_dict[(sbj,ses,\'card_regs\')] = slibase_df[selected_card_regs]\n                physio_reg_dict[(sbj,ses,\'resp_regs\')] = slibase_df[selected_resp_regs]\n            else:\n                physio_reg_dict[(sbj,ses,\'RVT_regs\')] = None\n                physio_reg_dict[(sbj,ses,\'card_regs\')] = None\n                physio_reg_dict[(sbj,ses,\'resp_regs\')] = None\n        else:\n            physio_reg_dict[(sbj,ses,\'RVT_regs\')] = None\n            physio_reg_dict[(sbj,ses,\'card_regs\')] = None\n            physio_reg_dict[(sbj,ses,\'resp_regs\')] = None\n    with open(filename, \'wb\') as f:\n        pickle.dump(physio_reg_dict, f)\n')


# ***
# 
# # Create Dashboard

# In[37]:


label_mapping = {r:r.replace(f'{CENSORING_MODE}_','') for r in pp_opts.values()}
label_mapping


# In[38]:


avial_qc_metrics = list(set([key[1] for key in QC_metrics.keys()]))

sbj_select                              = pn.widgets.Select(name='Subject',        options=sbj_list, width=200)
ses_select                              = pn.widgets.Select(name='Data Type',      options=ses_list+['all'], width=200)
pp_select                               = pn.widgets.Select(name='Pre-processing', options=pp_opts, width=200)
nordic_select                           = pn.widgets.Select(name='NORDIC',         options=nordic_opts, width=200)
fc_select                               = pn.widgets.Select(name='FC Metric',      options={'Correlation':'R','Covariance':'C'}, width=200)
plot_select                             = pn.widgets.Select(name='Plot type',      options={'Scatter Plot':'scatter','Hex Bin':'hexbin',
                                                                                            'Timeseries':'timeseries',
                                                                                            'FC Matrices across pipelines':'FCmats',
                                                                                            'FC Matrices across echoes':'FCmats_echoes',
                                                                                            'Group Results (Static)':'group_res_static', 'Group Results (Dynamic)':'group_res_dynamic'})

scat_lim_input                          = pn.widgets.FloatInput(name='Scatter Limit Value', value=1., step=0.1, start=0., end=50., width=200)
show_line_fit_checkbox                  = pn.widgets.Toggle(name='Show Linear Fit', button_type='primary')
scatter_extra_confs_card                = pn.Card(scat_lim_input,show_line_fit_checkbox, title='Scatter Plot & FCs | Configuration')

num_ics_to_show_input                   = pn.widgets.IntInput(name='Number of ICAs', value=3, width=200)
ts_confs_card                           = pn.Card(num_ics_to_show_input, title='Timeseries | Configuration')

qc_metric_select                        = pn.widgets.Select(name='QC Metric to show', options=avial_qc_metrics, value='pBOLD', width=200)
pps_to_include_in_group_results         = pn.widgets.MultiSelect(name='Pipelines to include in Group Results', options=pp_opts, value=list(pp_opts.values())[0:3], width=200)
remove_outliers_from_swarm_plots_toggle = pn.widgets.Toggle(name='Remove Outliers from BarPlot', button_type='primary')
show_stats_toggle                       = pn.widgets.Toggle(name='Show Statistical Annotations', button_type='primary')
show_points_toggle                      = pn.widgets.Toggle(name='Show Individual Points', button_type='primary')
stat_test_select                        = pn.widgets.Select(name='Statistical Test', options={'Paired T-test':'t-test_paired','Independent T-test':'t-test_ind','Mann Whitney (Ind,non-param)':'Mann-Whitney'})
annot_type_select                       = pn.widgets.Select(name='Annotation Type', options={'Stars':'star','Simple Annotation':'simple','Full Annotation':'full'})
barplot_extra_confs_card                = pn.Card(qc_metric_select, show_points_toggle, show_stats_toggle,
                                                  stat_test_select, annot_type_select, pps_to_include_in_group_results,
                                                  remove_outliers_from_swarm_plots_toggle, title='Group Results (Static) | Configuration')

sidebar = [sbj_select,ses_select,pp_select,nordic_select,fc_select, pn.layout.Divider(),
           plot_select,pn.layout.Divider(),
           scatter_extra_confs_card,pn.layout.Divider(),ts_confs_card,pn.layout.Divider(),
           barplot_extra_confs_card,
           ]


# In[39]:


@pn.depends(sbj_select,ses_select, pp_select, nordic_select, fc_select, plot_select, show_line_fit_checkbox, scat_lim_input,show_stats_toggle,stat_test_select,annot_type_select,pps_to_include_in_group_results,remove_outliers_from_swarm_plots_toggle, qc_metric_select,show_points_toggle,num_ics_to_show_input)
def get_main_frame(sbj,ses, pp, nordic, fc_metric, plot_type, show_line_fit, ax_lim,show_stats,stat_test,annot_type,pps_to_include_in_barplot,remove_outliers_from_swarm_plots, qc_metric,show_points,num_ics_to_show):
    if plot_type == 'timeseries':
        frame = get_ts_report_page(sbj,ses,pp,nordic,fc_metric,num_ics_to_show,
                       gs_df_dict,ica_dict,physio_dict,physio_reg_dict,pBOLD_xr,QC_metrics)
        return frame
    if plot_type == 'hexbin':
        frame = fc_across_echoes_scatter_page(DATASET,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, 
                                              data_fc,pBOLD_xr,QC_metrics,
                                              show_line=show_line_fit, ax_lim=ax_lim, hexbin=True)
        return frame
    if plot_type == 'scatter':
        frame = fc_across_echoes_scatter_page(DATASET,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, 
                                              data_fc,pBOLD_xr,QC_metrics,
                                              show_line=show_line_fit, ax_lim=ax_lim, hexbin=False)
        #frame = fc_across_echoes_scatter_page(DATASET,data_fc,pBOLD_xr,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, hexbin=False)
        return frame
    if plot_type == 'FCmats':
        fcR = get_fc_matrices(data_fc,pBOLD_xr,sbj,ses, nordic, 'R', net_cmap=power264_nw_cmap)
        fcC = get_fc_matrices(data_fc,pBOLD_xr,sbj,ses, nordic, 'C', net_cmap=power264_nw_cmap)
        return pn.Column(fcR,fcC)
    if plot_type == 'FCmats_echoes':
        layout = pn.GridBox(ncols=3)
        for ep in echo_pairs:
            fcR = get_fc_matrix(data_fc,pBOLD_xr,sbj,ses,pp,nordic,fc_metric,echo_pair=ep, net_cmap=power264_nw_cmap, ax_lim=ax_lim, title='%s | %s' % (fc_metric,ep))
            layout.append(fcR)
        return layout
    if plot_type == 'group_res_static':
        #data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        if ses == 'all':
            data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        else:
            idx = pd.IndexSlice
            data_to_show = QC_metrics[fc_metric,qc_metric].set_index(['Pre-processing','Session']).loc[idx[pps_to_include_in_barplot, ses], :].reset_index()
        aa = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  hue='Pre-processing',x='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left',show_points=show_points, session=ses, dot_size=1),    title=f'{qc_metric} grouped by NORDIC')
        bb = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  x='Pre-processing',hue='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left', show_points=show_points, session=ses, dot_size=1),   title=f'{qc_metric} grouped by Pre-processing')
        return pn.GridBox(*[aa,bb],ncols=2)
    if plot_type == 'group_res_dynamic':
        if fc_metric == 'C':
            pBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, fc_metric, 'pBOLD', nordic),title='pBOLD')
            return pn.Row(pBOLD_card,None,None)
        else:
            return pn.pane.Markdown('# This is not available for R-based FC')


# In[40]:


template = pn.template.BootstrapTemplate(title=f'{DATASET} Dataset | Edge-based Results', 
                                         sidebar=sidebar,
                                         main=get_main_frame)


# In[41]:


dashboard = template.show() #template.show(port=port_tunnel, open=False)


# In[42]:


dashboard.stop()


# ***

# In[44]:


from utils.dashboard import gen_scatter


# In[64]:


sbj,ses='sub-20','ses-2'
gen_scatter(DATASET,data_fc,sbj,ses,'ALL_Basic','off','e01|e02','e03|e03','C', show_linear_fit=False, ax_lim=None, hexbin=False, title=None, color='red') *\
gen_scatter(DATASET,data_fc,sbj,ses,'ALL_Basic','on','e01|e02','e03|e03','C', show_linear_fit=False, ax_lim=None, hexbin=False, title=None, color='green')


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[112]:


sbj,ses='sub-20','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))


# In[113]:


sbj,ses='sub-16','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))


# In[114]:


sbj,ses='sub-31','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))


# In[116]:


sbj,ses='sub-95','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
#ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
#ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))


# In[117]:


sbj,ses='sub-16','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


import holoviews as hv
def gen_scatter(dataset,data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_linear_fit=False, ax_lim=None, hexbin=False, title=None, color='blue'):
    """
    Generate scatter plot for two different FC matrices

    Inputs:
    -------
    dataset (str): name of the dataset being used (e.g., evaluation, discovery)
    data_fc (dict): dictionary with FC matrices
    sbj (str): subject ID
    ses (str): session ID
    pp (str): pre-processing pipeline
    nordic (str): whether Nordic was used or not
    eep1 (str): first echo pair in the format 'e02|e02'
    eep2 (str): second echo pair in the format 'e02|e02'
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    show_linear_fit (bool): whether to show linear fit line or not
    ax_lim (float): axis limits for the scatter plot
    hexbin (bool): whether to use hexbin plot instead of scatter plot

    Returns:
    --------
    hvplot object with the scatter plot and theoretical lines

    """
    echo_times_dict = TES_MSEC[dataset]

    if (sbj,ses,pp,nordic,eep1,fc_metric) not in data_fc:
        return pn.pane.Markdown('#Not Available')
    data_df = pd.DataFrame(columns=[eep1,eep2, fc_metric])
    data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)
    data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)
    # Compute limits for X and Y axis
    if ax_lim is None:
        if fc_metric == 'R':
            lims = (-1,1) 
        else:
            lims = (data_df.quantile(0.01).min(),data_df.quantile(0.99).max())
    else:
        lims=(-ax_lim,ax_lim)
    # Create scatter plot and fitted line
    scat           = data_df.hvplot.scatter(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7, title=title, color=color) #.opts(active_tools=['save'], tools=['save'])
    data_lin_fit   = hv.Slope.from_scatter(scat).opts(line_width=3, line_color='#0f0fff') #.opts(active_tools=['save'], tools=['save'])

    if hexbin:
        scat = data_df.hvplot.hexbin(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7)
    # Compute theoretical slopes for extreme BOLD and So dominated regimes
    if fc_metric  == 'R':
        BOLD_line_sl, BOLD_line_int = 1.,0.
    else:
        e1_X,e2_X     = eep1.split('|')
        e1_Y,e2_Y     = eep2.split('|')
        BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])
        BOLD_line_int = 0.
    
    So_line_sl, So_line_int = 1.,0.
    
    # Create Theoretical BOLD and So lines
    BOLD_line  = hv.Slope(BOLD_line_sl, BOLD_line_int).opts(line_color='g',line_width=2, line_dash='dashed') #.opts(active_tools=['save'], tools=['save'])
    So_line    = hv.Slope(So_line_sl,   So_line_int  ).opts(line_color='r',line_width=2, line_dash='dashed') #.opts(active_tools=['save'], tools=['save'])
    
    # Join all graphical elements
    if show_linear_fit:
        plot = (scat * data_lin_fit * So_line * BOLD_line) #.opts(toolbar=None)
    else:
        plot = (scat * So_line * BOLD_line)
     
    return plot.opts(active_tools=['reset'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


import holoviews as hv
data_qc = QC_metrics
resp_plot, card_plot = None,None
if (sbj,ses) in data_qc['C',('Physio (resp)')].index:
    resp_plot = data_qc['C',('Physio (resp)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Card. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
    resp_plot = resp_plot * hv.Points((data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
    resp_plot = resp_plot.opts(shared_axes=False)
if (sbj,ses) in data_qc['C',('Physio (cardiac)')].index:
    card_plot = data_qc['C',('Physio (cardiac)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Resp. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
    card_plot = card_plot * hv.Points((data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
    card_plot = card_plot.opts(shared_axes=False)

physio_card = pn.Card(pn.Column(resp_plot,card_plot),title='Physio',width=350)
physio_card


# In[58]:


accuracy_dict['Pearson'].set_index(['Session']).loc['ses-1'].reset_index(drop=True)


# In[41]:


dashboard.stop()


# ***
# 
# # Explore extreme cases
# ## a) Worse pBOLD scan

# In[119]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[121]:


mms = MinMaxScaler(feature_range=(2, 100))
motion_df = pd.DataFrame(index=ds_index, columns = ['Max. Motion (enorm)','Mean Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max. Motion (enorm)'] = aux_mot.max()
motion_df['Mean Motion (dot size)'] = mms.fit_transform(motion_df['Mean Motion (enorm)'].values.reshape(-1,1))
motion_df['Max. Motion (dot size)'] = mms.fit_transform(motion_df['Max. Motion (enorm)'].values.reshape(-1,1))
motion_df = motion_df.infer_objects()
motion_val = motion_df.loc[sbj,ses].values[0]


# In[122]:


pBOLD_df = QC_metrics['C','pBOLD']
pBOLD_df = pBOLD_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)


# In[123]:


TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)


# In[ ]:





# In[143]:


df = pd.concat([TSNR_df.set_index(['Subject','Session']),pBOLD_df.set_index(['Subject','Session']), motion_df],axis=1)
cbar_min = df['Max. Motion (enorm)'].quantile(0.05)
cbar_max = df['Max. Motion (enorm)'].quantile(0.99)

df.hvplot.scatter(y='TSNR (Full Brain)',x='pBOLD', hover_cols=['Subject','Session'],aspect='square',s='Max. Motion (dot size)',c='Max. Motion (enorm)', cmap='cividis', fontscale=1.5, frame_width=350).opts(clim=(cbar_min,cbar_max),colorbar_opts={'title':'Max. Motion (mm):'})


# In[125]:


df


# In[152]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set seaborn theme
sns.set(style="white", context="notebook")

# Compute quantile limits for color normalization
vmin = df["Max. Motion (enorm)"].quantile(0.05)
vmax = df["Max. Motion (enorm)"].quantile(0.99)

# Create JointGrid
g = sns.JointGrid(data=df, x="pBOLD", y="TSNR (Full Brain)", space=0, height=8)

# Scatter plot with color and size
df = df[["pBOLD", "TSNR (Full Brain)", "Max. Motion (dot size)", "Max. Motion (enorm)"]].dropna()

#sizes = df["Max. Motion (dot size)"]
#scaled_sizes = 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min()) + 10  # Scale between 10 and 210
sc = g.ax_joint.scatter(
    data=df,
    x="pBOLD",
    y="TSNR (Full Brain)",
    s=df['Max. Motion (dot size)'],
    c=df["Max. Motion (enorm)"],
    cmap="cividis",
    vmin=vmin, vmax=vmax,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.3
)

# Create new axes above the top marginal histogram
cbar_ax = g.fig.add_axes([pos_joint.x0, pos_marg_x.y1 + 0.05, pos_joint.width, 0.02])  # [left, bottom, width, height]

# Add horizontal colorbar
cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')

# Move label and ticks to the top
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.tick_top()
cbar.set_label("Max. Motion (mm)", rotation=0, labelpad=5)

# Top and right marginal plots
sns.histplot(x=df["pBOLD"], ax=g.ax_marg_x, kde=True, color="#72B6A1", edgecolor="black", bins=50)
sns.histplot(y=df["TSNR (Full Brain)"], ax=g.ax_marg_y, kde=True, color="#72B6A1", edgecolor="black", bins=50)

# Axis labels
g.set_axis_labels("$p_{BOLD}$", "TSNR (Full Brain)")

# Improve layout
plt.tight_layout()
plt.show()


# In[86]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:


sbj,ses,pBOLD_val = pBOLD_df.reset_index(drop=True).sort_values(by='pBOLD',ascending=True).iloc[0]
TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
TSNR_val = TSNR_df.set_index(['Subject','Session']).loc[sbj,ses].values[0]
(sbj,ses)


# In[151]:


motion_df = pd.DataFrame(index=ds_index, columns = ['Max Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        #motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_val = motion_df.loc[sbj,ses].values[0]


# In[152]:


fig,ax = plt.subplots(1,3, figsize=(10,5))
sns.barplot(data=pBOLD_df, alpha=0.5,ax=ax[0])
sns.swarmplot(data=pBOLD_df,s=3,ax=ax[0])
ax[0].axhline(pBOLD_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=TSNR_df, alpha=0.5,ax=ax[1])
sns.swarmplot(data=TSNR_df,s=2,ax=ax[1])
ax[1].axhline(TSNR_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=motion_df, ax=ax[2], alpha=0.5)
sns.swarmplot(data=motion_df,s=2,ax=ax[2])
ax[2].axhline(motion_val, color='k',linestyle='dashed', linewidth=0.5)


# ## b) Second Worse TSNR scan

# In[153]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[157]:


pBOLD_df = QC_metrics['C','pBOLD']
pBOLD_df = pBOLD_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
sbj,ses,pBOLD_val =pBOLD_df.reset_index(drop=True).sort_values(by='pBOLD',ascending=True).iloc[1]
TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
TSNR_val = TSNR_df.set_index(['Subject','Session']).loc[sbj,ses].values[0]
(sbj,ses)


# In[156]:


fig,ax = plt.subplots(1,3, figsize=(10,5))
sns.barplot(data=pBOLD_df, alpha=0.5,ax=ax[0])
sns.swarmplot(data=pBOLD_df,s=3,ax=ax[0])
ax[0].axhline(pBOLD_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=TSNR_df, alpha=0.5,ax=ax[1])
sns.swarmplot(data=TSNR_df,s=2,ax=ax[1])
ax[1].axhline(TSNR_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=motion_df, ax=ax[2], alpha=0.5)
sns.swarmplot(data=motion_df,s=2,ax=ax[2])
ax[2].axhline(motion_val, color='k',linestyle='dashed', linewidth=0.5)


# ## b) Third Worse TSNR scan

# In[153]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[158]:


pBOLD_df = QC_metrics['C','pBOLD']
pBOLD_df = pBOLD_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
sbj,ses,pBOLD_val =pBOLD_df.reset_index(drop=True).sort_values(by='pBOLD',ascending=True).iloc[2]
TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
TSNR_val = TSNR_df.set_index(['Subject','Session']).loc[sbj,ses].values[0]
(sbj,ses)


# In[159]:


motion_df = pd.DataFrame(index=ds_index, columns = ['Max Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        #motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_val = motion_df.loc[sbj,ses].values[0]


# In[160]:


fig,ax = plt.subplots(1,3, figsize=(10,5))
sns.barplot(data=pBOLD_df, alpha=0.5,ax=ax[0])
sns.swarmplot(data=pBOLD_df,s=3,ax=ax[0])
ax[0].axhline(pBOLD_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=TSNR_df, alpha=0.5,ax=ax[1])
sns.swarmplot(data=TSNR_df,s=2,ax=ax[1])
ax[1].axhline(TSNR_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=motion_df, ax=ax[2], alpha=0.5)
sns.swarmplot(data=motion_df,s=2,ax=ax[2])
ax[2].axhline(motion_val, color='k',linestyle='dashed', linewidth=0.5)


# In[130]:


motion_df


# In[ ]:




