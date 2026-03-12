#!/usr/bin/env python
# coding: utf-8

# # 1. Setup and Study Configuration
# 
# This notebook prepares all inputs and batch commands needed to run Connectome-based Predictive Modeling (CPM) analyses.
# 
# The setup code below:
# - imports required libraries and project utilities,
# - captures the current username for user-specific swarm/log paths,
# - defines dataset, atlas, pipeline, and CPM parameters,
# - loads dataset/atlas indexing metadata,
# - computes scan-level motion summaries used as confounds.
# 

# In[33]:


import os.path as osp
import os
import pandas as pd
import numpy as np
from utils.basics import get_dataset_index, get_altas_info
from utils.basics import PRCS_DATA_DIR, CODE_DIR, ATLASES_DIR, DOWNLOAD_DIRS, PRJ_DIR
import pickle
import datetime
from pathlib import Path
from tqdm.notebook import tqdm
from nilearn.connectome import sym_matrix_to_vec


# In[23]:


import getpass
username = getpass.getuser()
print(username)


# In[ ]:


DATASET='evaluation' # Dataset used in this part of the analyses
PIPELINES = ['ALL_Basic','ALL_GS','ALL_Tedana-fastica'] # Pipelines being evaluated in this part of the analyses

ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]

# CPM Configuration
BEHAVIOR  = 'nihcog_fluidcomp'
CORR_MODE = 'spearman'


# In[3]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# Load Atlas Information

# In[4]:


roi_info_df, power264_nw_cmap = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)


# Load Head Motion Estimates

# In[5]:


motion_df = pd.DataFrame(index=ds_index, columns = ['Max. Motion (enorm)','Mean Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max. Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_df.head(3)


# # 2. Prepare FC Predictors for CPM
# 
# ## 2.1 Load precomputed FC matrices
# 
# Read the precomputed FC dictionary from disk. Entries are keyed by subject, session, preprocessing pipeline, NORDIC status, echo pairing, and connectivity mode.
# 

# In[6]:


fc_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_FC.pkl')
print('++ FC will be loaded from %s' % fc_path)


# In[7]:


get_ipython().run_cell_magic('time', '', "with open(fc_path, 'rb') as f:\n        data_fc = pickle.load(f)\n")


# CPM will only be run on subjecs with two scans, so we can compare the reliability of results

# In[8]:


# Grab subjects with two scans available
sbjs_with_two_scans = {'ses-1':[],'ses-2':[],'both':[]}
for sbj in sbj_list:
    if ((sbj,'ses-1','ALL_Basic','off','e02|e02','R') in data_fc.keys()):
        sbjs_with_two_scans['ses-1'].append(sbj)
    if ((sbj,'ses-2','ALL_Basic','off','e02|e02','R') in data_fc.keys()):
        sbjs_with_two_scans['ses-2'].append(sbj)
    if ((sbj,'ses-1','ALL_Basic','off','e02|e02','R') in data_fc.keys()) & ((sbj,'ses-2','ALL_Basic','off','e02|e02','R') in data_fc.keys()):
        sbjs_with_two_scans['both'].append(sbj)
print("++ INFO: Number of subjects with imaging data available for session 1: %d" % len(sbjs_with_two_scans['ses-1']))
print("++ INFO: Number of subjects with imaging data available for session 2: %d" % len(sbjs_with_two_scans['ses-2']))
print("++ INFO: Number of subjects with imaging data available for sessions 1 and 2: %d" % len(sbjs_with_two_scans['both']))


# ## 2.2 Restrict to analyzable scans and vectorize FC
# 
# For CPM, we use middle-echo (`e02|e02`) and R-based connectivity. We keep subjects with available imaging per session, then convert each symmetric FC matrix into its upper-triangle feature vector (unique edges only).
# 

# In[9]:


fc_data = {}
for pp in PIPELINES:
    for NORDIC in ['off','on']:
        for ses in ['ses-1','ses-2']:
            fc_data[(ses,pp,NORDIC)] = pd.DataFrame(index=sbjs_with_two_scans[ses],columns=range(Ncons))
            for sbj in tqdm(sbjs_with_two_scans[ses],desc='%s | %s | %s' % (pp,NORDIC,ses)):
                aux                               = sym_matrix_to_vec(data_fc[(sbj,ses,pp,NORDIC,'e02|e02','R')].values, discard_diagonal=True)
                fc_data[(ses,pp,NORDIC)].loc[sbj] = aux
            fc_data[(ses,pp,NORDIC)].columns = range(Ncons) # Needed to ensure the proper type on the column ids


# # 3. Load Behavioral Target
# 
# Load the target behavior (`nihcog_fluidcomp`) from the behavioral table, align it to the retained subject sets per session, and drop missing values.
# 

# In[13]:


path       = osp.join(DOWNLOAD_DIR,'ddbehav_updated_20230530.csv')
behav_data, Nbehavs = {},{}
for ses in ['ses-1','ses-2']:
    df         = pd.read_csv(path)
    df         = df.set_index('id').loc[sbjs_with_two_scans[ses],BEHAVIOR]
    df         = df.reset_index()
    df.index   = df['id'].values
    behav_data[ses] = df.drop(['id'],axis=1)
    behav_data[ses] = behav_data[ses].dropna()
    Nbehavs[ses]    = behav_data[ses].shape[1]
    print('++ [%s] Number of subjects with behavioral data available: %d' % (ses,behav_data[ses].shape[0]))


# # 4. Export CPM Inputs to Cache
# 
# Create a clean cache directory for CPM inputs. The following cells save behavior targets, FC feature matrices, and motion confounds in the format expected by the batch CPM scripts.
# 

# In[16]:


CPM_CACHE_DIR=osp.join(CODE_DIR,'notebooks','cache','cpm')
if not osp.exists(CPM_CACHE_DIR):
    os.makedirs(CPM_CACHE_DIR)
else:
    rmtree(CPM_CACHE_DIR)
    os.makedirs(CPM_CACHE_DIR)
print(CPM_CACHE_DIR)


# ## 4.1 Save target behavior tables
# 
# Persist one behavior table per session so each CPM run can load the correct prediction target.
# 

# In[17]:


for ses in ['ses-1','ses-2']:
    behav_out_path = osp.join(CPM_CACHE_DIR,f'{ses}_behavior.pkl')
    behav_data[ses].to_pickle(behav_out_path)
    print ('++ INFO [main]: Behaviors table loaded into memory [#Behaviors=%d] --> %s' % (Nbehavs[ses],behav_out_path))


# ## 4.2 Save FC feature matrices
# 
# For each scenario (session x pipeline x NORDIC), keep only subjects with valid behavior data, verify index alignment, and save FC predictors.
# 

# In[47]:


for scenario in fc_data.keys():
    ses,pp,NORDIC = scenario
    fc_data[scenario]    = fc_data[scenario].loc[behav_data[ses].index]
    assert fc_data[scenario].index.equals(behav_data[ses].index), "++ ERROR [main]:Index in FC dataFrame [%s] and behavior dataframe do not match." % scenario
    fc_out_path = osp.join(CPM_CACHE_DIR,f'{ses}_{pp}_NORDIC-{NORDIC}_fc.pkl')
    fc_data[scenario].to_pickle(fc_out_path)
    print('++ INFO [main]: FC data loaded into memory [%s] | Number of scans = %d --> %s' % (scenario,fc_data[scenario].shape[0],fc_out_path))


# ## 4.3 Save confound tables
# 
# Save session-specific motion confounds for the same subjects used in prediction. These confounds are used for orthogonalization before model fitting.
# 

# In[48]:


for ses in ['ses-1','ses-2']:
    this_motion_df = motion_df.loc[list(behav_data[ses].index),:]
    confounds      = this_motion_df.xs(ses, level='Session')
    confounds_out_path = osp.join(CPM_CACHE_DIR,f'{ses}_motion.pkl')
    confounds.to_pickle(confounds_out_path)


# # 5. Build Batch Infrastructure for CPM Runs
# 
# ## 5.1 Real-data CPM jobs
# 
# Set paths for swarm scripts and logs used to launch CPM across sessions, preprocessing pipelines, and NORDIC settings.
# 

# In[49]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N13_CPM_IQ.{ATLAS_NAME}.SWARM.sh')
print('++ INFO: Swarm script for CPM (real data): %s' % script_path)


# ## 5.2 Configure log directory
# 
# Create (if needed) the directory that will store stdout/stderr logs from all CPM swarm jobs.
# 

# In[29]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N13_CPM_IQ.{ATLAS_NAME}.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print('++ INFO: Folder for log files related to CPM (real data): %s' % log_path)


# ## 5.3 Generate swarm commands
# 
# Write one swarm command per CPM model configuration (2 sessions x 3 pipelines x 2 NORDIC settings x 100 iterations), passing behavior, FC predictors, confounds, and runtime options to `cpm_batch.sh`.
# 

# In[ ]:


with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for ses in ['ses-1','ses-2']:
        for pp in ['ALL_Basic','ALL_GS','ALL_Tedana-fastica']:
            for NORDIC in ['off','on']:
                for num_iter in range(100):
                    program_path          = osp.join(CODE_DIR,'python','cpm_batch.sh')
                    behavior_pickle_file  = osp.join(CODE_DIR,'notebooks','cache','cpm',f'{ses}_behavior.pkl')
                    output_path           = osp.join(CODE_DIR,'notebooks','cache','cpm')
                    fc_pickle_file        = osp.join(CODE_DIR,'notebooks','cache','cpm',f'{ses}_{pp}_NORDIC-{NORDIC}_fc.pkl')
                    confounds_pickle_file = osp.join(CODE_DIR,'notebooks','cache','cpm',f'{ses}_motion.pkl')
                    the_file.write(f'export BEHAV_FILE={behavior_pickle_file} BEHAV=nihcog_fluidcomp OUT_LABEL={ses}_{pp}_NORDIC_{NORDIC} OUT_DIR={output_path} ITER={num_iter} FC_PATH={fc_pickle_file} P_THR=0.01 COR_MODE={CORR_MODE} CONFOUND_FILE={confounds_pickle_file} OTHER_PARS="-C -v"; sh {program_path} \n')
print(script_path)


# ## 5.4 Quick output sanity check
# 
# Count files in each CPM output folder to quickly verify how many iteration outputs currently exist per scenario.
# 

# In[56]:


for ses in ['ses-1','ses-2']:
    for pp in ['ALL_Basic','ALL_GS','ALL_Tedana-fastica']:
        for NORDIC in ['off','on']:
            output_path = osp.join(CODE_DIR,'notebooks','cache','cpm',f'{ses}_{pp}_NORDIC_{NORDIC}')
            output_path = Path(output_path)
            files  = [item for item in output_path.iterdir() if item.is_file()]
            file_count = len(files)
            print('++ [%s | %s | %s] --> %d outputs found' % (ses, pp, NORDIC, file_count))


# In[ ]:




