#!/usr/bin/env python
# coding: utf-8

# # Description: Computation of pBOLD
# 
# This notebook creates the SWARM file to compute pBOLD in all scans of a given dataset.
# 
# The user must input the dataaset (evalution or discovery) when prompted in cell 2
# 
# The swarm file must then be submitted to the cluster
# 
# The last cell allows to identify potential jobs that might have failed.

# In[1]:


import os.path as osp
import os
from utils.basics import get_dataset_index
from utils.basics import PRJ_DIR, ATLASES_DIR, CODE_DIR, TES_MSEC, PIPELINES
import datetime
from tqdm.notebook import tqdm


# In[2]:


DATASET = input ('Select Dataset (discovery or evaluation):')


# In[3]:


ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR  = osp.join(ATLASES_DIR,ATLAS_NAME)

echo_times_dict = TES_MSEC[DATASET]


# Get username. We use this to create user specific locations for SWARM and log files.

# In[4]:


import getpass
username = getpass.getuser()
print(username)


# Get list of scans in the dataset

# In[5]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# # 2. Create Swarm Script to Extract ROI TS from Basic denoised data
# 
# Create path to SWARM script

# In[6]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N05_Compute_pBOLD.{DATASET}.{ATLAS_NAME}.SWARM.sh')
print(script_path)


# Create folder for logs created by the batch jobs

# In[7]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N05_Compute_pBOLD.{DATASET}.{ATLAS_NAME}.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)


# Create the SWARM script. This script will have one line per scan

# In[8]:


echo_times_in_msec = ','.join([str(te) for _,te in echo_times_dict.items()])


# In[9]:


with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 30 --time 00:05:00 --logdir {log_path} --partition quick,norm\n')
    the_file.write('\n')
    for sbj,ses in list(ds_index):
        for NORDIC in ['on','off']:
            for fc_metric in ['corr','cov']:
                for pp in PIPELINES:
                    e01_ts_path = osp.join(PRJ_DIR,'prcs_data',sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}',f'errts.{sbj}.r01.e01.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    e02_ts_path = osp.join(PRJ_DIR,'prcs_data',sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}',f'errts.{sbj}.r01.e02.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    e03_ts_path = osp.join(PRJ_DIR,'prcs_data',sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}',f'errts.{sbj}.r01.e03.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    out_path    = osp.join(PRJ_DIR,'prcs_data',sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}',f'errts.{sbj}.r01.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.pBOLD_{fc_metric}.csv')
                    the_file.write(f'export E01_TS_PATH={e01_ts_path}  E02_TS_PATH={e02_ts_path} E03_TS_PATH={e03_ts_path} TE_LIST={echo_times_in_msec} METRIC={fc_metric} OUT_PATH={out_path}; sh {CODE_DIR}/bash/compute_pBOLD.sh \n')
the_file.close()     
print(f'Swarm script written to: {script_path}')


# # 3. Check all outputs were created

# In[10]:


needed_outputs = []
missing_outputs = []
for sbj,ses in tqdm(list(ds_index)):
    for NORDIC in ['on','off']:
        for fc_metric in ['corr','cov']:
            for pp in PIPELINES:
                out_path    = osp.join(PRJ_DIR,'prcs_data',sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}',f'errts.{sbj}.r01.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.pBOLD_{fc_metric}.csv')
                needed_outputs.append(out_path)
                if not osp.exists(out_path):
                    missing_outputs.append(out_path)
print(" Missing %d files of %d needed" %(len(missing_outputs), len(needed_outputs)))


# In[ ]:




