#!/usr/bin/env python
# coding: utf-8

# # Description: Generation and evaluation of physio regressors contribution to the GS
# 
# This notebook will do the following:
# 
# 1) Attempt to run AFNI program ```physio_calc``` on scans for which physiological timeseries are available.
# 2) Automaticall detect a subset of scans where ```physio_calc``` has done its job correctly
# 3) Compute the variance explained in the global signal by physiological regressors
# 4) Create a null distribution of variance explained

# In[1]:


import os.path as osp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from utils.basics import PRJ_DIR, PRCS_DATA_DIR, DOWNLOAD_DIRS, CODE_DIR, read_group_physio_reports
from sklearn.ensemble import IsolationForest
import hvplot.pandas
import holoviews as hv
from scipy.stats import zscore
import statsmodels.api as sm
from afnipy.lib_afni1D import Afni1D
import shutil
import pickle
from utils.basics import get_dataset_index


# In[2]:


import getpass
username = getpass.getuser()
print(username)


# We only looked at the physiological data in the ```evaluation``` dataset

# In[3]:


DATASET = 'evaluation'
DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]


# Get list of fMRI scans from the Spreng Dataset that passed our intial QC and are included in this study

# In[4]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# ***
# # 1. Run ```physio_calc``` in all scans with physio data available
# 
# We will do this in biowulf using batch jobs. Here we create the necessary infrastructure for that
# 
# ### 1.1. Create Swarm path

# In[5]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N06a_Compute_Physio_Regressors.SWARM.sh')
print(script_path)


# ### 1.2. Create folder for log files

# In[6]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N06a_Compute_Physio_Regressors.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)


# ### 1.3. Write Swarm File
# 
# This file will contain one line per scan (sbj,ses) for which we were able to find both ```{sbj}_{ses}_task-rest_physio.tsv.gz``` and ```{sbj}_{ses}_task-rest_physio.json``` files

# In[7]:


no_physio_scans = []

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for sbj,ses in tqdm(ds_index):
        physio_path = osp.join(DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.tsv.gz')
        json_path   = osp.join(DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.json')
        if osp.exists(physio_path) and osp.exists(json_path):
            for ec in [1,2,3]:
                dset_path = osp.join(DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_echo-{ec}_bold.nii.gz')
                out_dir = osp.join(PRCS_DATA_DIR,sbj,'D04_Physio')
                prefix = f'{sbj}_{ses}_task-rest_echo-{ec}'
                the_file.write(f'physio_calc.py -phys_file {physio_path} -phys_json {json_path} -dset_epi {dset_path} -out_dir {out_dir} -prefix {prefix} -prefilt_mode median -prefilt_max_freq 50\n')
        else:
            no_physio_scans.append((sbj,ses))    
the_file.close()     


# ### 1.4 Run batch jobs in biowulf
# 
# ```bash
# 
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N06a_Compute_Physio_Regressors.SWARM.sh -g 8 -t 8 -b 20 --time 00:10:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N06a_Compute_Physio_Regressors.log --partition quick,norm --module afni
# ```

# ***
# # 2. Check the outputs of ```phys_calc```
# 
# For some scans, even though the physio files are available, they do not contain a sufficient amount of samples. In those cases ```physio_calc``` cannot run.
# 
# We next try to identify those instances and come with a final list of scans for which ```physio_calc``` completed. 
# 
# ### 2.1. Get list of scans with existing automatically computed physiological regressors
# 
# First, we get the list of scans in which we attempted to run physio_calc

# In[8]:


sbj_ses_with_physio = ds_index.drop(no_physio_scans)
len(sbj_ses_with_physio)


# Second, we check for which of these scans there is an ```slibase``` file. This is the primary output of ```phys_calc``` that contains the RVT and RETROICOR regressors.

# In[9]:


sbj_ses_physio_corrupted = []
for sbj,ses in tqdm(sbj_ses_with_physio):
    for ec in [1,2,3]:
        file_path = osp.join(PRCS_DATA_DIR,sbj,'D04_Physio',f'{sbj}_{ses}_task-rest_echo-{ec}_slibase.1D')
        if not osp.exists(file_path):
            sbj_ses_physio_corrupted.append((sbj,ses))


# In[10]:


sbj_ses_physio_corrupted = list(set(sbj_ses_physio_corrupted))
print('++ INFO: Number of scans with physio available, but somehow corrupted: %d scans' % len(sbj_ses_physio_corrupted))
print(sbj_ses_physio_corrupted)


# Finally, we create a new list that only contains the scans for which ```physio_calc``` was able to generate a ```slibase``` file.

# In[11]:


scans_with_complete_physio = ds_index.drop(no_physio_scans + sbj_ses_physio_corrupted)
print("++ INFO: Number of scans for which we were able to complete physio_calc = %d scans" % len(scans_with_complete_physio))


# In[12]:


selected_scans = scans_with_complete_physio
print("++ INFO: Number of scans with complete physiological regressors: %d scans" % len(selected_scans))


# In[13]:


pd.DataFrame(index=selected_scans).reset_index().to_csv(f'./summary_files/{DATASET}_CompletePhysio_ScanList.csv', index=False)


# ***
# 
# # 3. Compute variance explained in the GS for physiological regressors
# 
# For the scans that we know have good physio regressors, we will now compute how much variance of the global signal can be explained by the physio regressors. This is done via batch jobs that call program ```GS_physio_exp_var```. The way this program estimates variance explained is as follows:
# 
# 1. Load provided global signal and physiological regressors into memory
# 2. Remove constant and linear trends from all loaded timeseries separately
# 3. For each regressor type (e.g., rvt01, rtv02, card.c1, etc) it finds the time-shifted version that most strongly correlates with the global signal. At the end of this step, we will have a list of 13 regressors.
# 4. Computes the variance explained by these 13 regressors.
# 
# ### 3.1. Create path for Swarm file

# In[14]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N06b_Compute_varexp_in_GS_by_physio.SWARM.sh')
print(script_path)


# ### 3.2. Create folder for logs

# In[15]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N06b_Compute_varexp_in_GS_by_physio.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)


# ### 3.3. Write Swarm file
# 
# This will contain one line per-scan that we have marked as having good physiological data.

# In[16]:


with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 5 --time 00:45:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for sbj,ses in tqdm(selected_scans):
        gs_path      = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.1D') #f'pb03.{sbj}.r01.e02.volreg.spc.GS.1D')
        slibase_path = osp.join(PRCS_DATA_DIR,sbj,'D04_Physio',f'{sbj}_{ses}_task-rest_echo-2_slibase.1D')
        output_path  = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl') #f'pb03.{sbj}.r01.e02.volreg.spc.GS.PhysioModeling.pkl')
        the_file.write(f'export GS_PATH={gs_path} PHYSIO_PATH={slibase_path} OUTPUT_PATH={output_path}; sh {CODE_DIR}/python/GS_physio_exp_var.sh\n')
the_file.close()     


# The next cell help us look for issues when running the batch jobs. If all things went well there should be no WARNING lines printed out.

# In[17]:


for sbj,ses in tqdm(selected_scans):
    output_path  = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
    if not osp.exists(output_path):
        print("++ WARNING: %s is missing" % output_path)


# We will now compile all results into a single csv file for later exploration

# In[18]:


df = pd.DataFrame(index=selected_scans,columns=['Var. Exp. by Physio Regressors'])
for sbj,ses in tqdm(selected_scans):
    # Variance Explained by Regressors
    model_path  = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
    with open(model_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    model = loaded_dict['model'] #sm.load(model_path)
    df.loc[(sbj,ses),'Var. Exp. by Physio Regressors'] = model.rsquared
df=df.infer_objects()
df.to_csv(f'./summary_files/{DATASET}_varexp_gs_physio.real_data.csv')


# ***
# 
# # 4. Create a NULL DISTRIBUTION for estimates of variance explained.
# 
# Both the global signal and the physiological regressors have quite constrained spectral characteristics. Moreover, to ensure we do not understimate how much physiology can explain the global signal, we are picking the best time-shited version of each regressor. Although those are good things to make sure we do not understimate the contribution of cardiac and respiratory function to the global signal, it can lead to over estimation. By generating a null distribution were we compute the variance explain in the global signal of one scan by the physiological regressors of another scan, we build a null distribution so that we can better contextualize our variance explained estimates.
# 
# ### 4.1. Create path for swarm file

# In[19]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N06c_Compute_varexp_in_GS_by_physio_nulls.SWARM.sh')
print(script_path)


# ### 4.2. Create folder for logs

# In[ ]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N06c_Compute_varexp_in_GS_by_physio_nulls.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)


# ### 4.3. Create a folder where to save the results of each of the 10,000 null permutations

# In[ ]:


perm_dir = osp.join(CODE_DIR,'notebooks','cache','gs_phys_varex_perms')
if osp.exists(perm_dir):
    shutil.rmtree(perm_dir)
os.makedirs(perm_dir)


# ### 4.4. Write the Swarm file
# 
# Here, for each permutation, we first randomly select one scan (sbj,ses) for the global signal. Then we randomly select one scan from any other subject for the physiological regressors.

# In[ ]:


n_null_cases = 10000
selected_scans_df = pd.DataFrame(index=selected_scans)
with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 10 --time 00:20:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for i in tqdm(range(n_null_cases)):
        ii = str(i).zfill(5)
        gs_sbj, gs_ses = selected_scans_df.sample(1).index.values[0]
        ph_sbj, ph_ses = pd.DataFrame(index=selected_scans.drop(gs_sbj,level='Subject')).sample(1).index.values[0]
        gs_path        = osp.join(PRCS_DATA_DIR,gs_sbj,f'D03_Preproc_{gs_ses}_NORDIC-off',f'pb03.{gs_sbj}.r01.e02.volreg.GS.1D')
        ph_path        = osp.join(PRCS_DATA_DIR,ph_sbj,'D04_Physio',f'{ph_sbj}_{ph_ses}_task-rest_echo-2_slibase.1D')
        out_path       = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
        the_file.write(f'export GS_PATH={gs_path} PHYSIO_PATH={ph_path} OUTPUT_PATH={out_path}; sh {CODE_DIR}/python/GS_physio_exp_var.sh\n')
the_file.close()     


# ### 4.5. Check all permutations finished correctly

# In[ ]:


for i in tqdm(range(n_null_cases)):
    ii = str(i).zfill(5)
    output_path  = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
    if not osp.exists(output_path):
        print("++ WARNING: %s is missing" % output_path)


# We will now compile all results into a single csv file for later exploration

# In[ ]:


df = pd.DataFrame(index=range(n_null_cases),columns=['Var. Exp. by Physio Regressors (NULL)'])
for i in tqdm(range(n_null_cases)):
    ii = str(i).zfill(5)
    model_path  = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
    with open(model_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    model = loaded_dict['model'] #sm.load(model_path)
    df.loc[i,'Var. Exp. by Physio Regressors (NULL)'] = model.rsquared_adj
df=df.infer_objects()
df.index.name='Permutation'
df.to_csv(f'./summary_files/{DATASET}_varexp_gs_physio.null_distribution.csv')

