#!/usr/bin/env python
# coding: utf-8

# # Description: Compute Kappa and Rho for Global Signal
# 
# In this notebook we estimate the kappa and the rho of the global signal

# In[1]:


import os.path as osp
import pandas as pd
import os
from tqdm import tqdm
import datetime
from utils.basics import PRJ_DIR, PRCS_DATA_DIR, CODE_DIR
from utils.basics import get_dataset_index


# In[2]:


import getpass
username = getpass.getuser()
print(username)


# In[3]:


DATASET = 'evaluation'


# In[4]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# # 1. Create batch jobs for computing kappa and rho for all scans
# 
# We will use the ```generate_metrics``` function within ```tedana``` to compute the kappa and the rho
# 
# ## 1.1. Create the path to the swarm file

# In[5]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N07_Compute_GS_kappa_and_rho.SWARM.sh')
print(script_path)


# ## 1.2. Create a folder for log files

# In[6]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N07_Compute_GS_kappa_and_rho.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)


# ## 1.3. Write the swarm file
# 
# This file will contain one line per scan and calls the program ```GS_kappa_and_rho.py```.

# In[7]:


with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for sbj,ses in tqdm(ds_index):
        the_file.write(f'export SBJ={sbj} RUN={ses} CENSOR_MODE=ALL; cd {CODE_DIR}/python/; sh {CODE_DIR}/python/GS_kappa_and_rho.sh \n')
the_file.close()     


# ***
# # 2. Check all jobs finalized correctly
# 
# As we do that, we will also create a .csv file ```evaluation_gs_kappa_rho.csv``` with the estimated kappa and rho for all scans. 
# 
# > **NOTE:** If you see any warnings, that means that the associated batch job did not finish correctly. Try again.
# 
# We will be able to load that in other notebooks.

# In[8]:


kappa_rho_df = pd.DataFrame(index=ds_index,columns=['kappa (GS)','rho (GS)','kappa_rho_color'])
for sbj,ses in tqdm(ds_index, desc='Scans:'):
    path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'{sbj}_{ses}_GS_kappa_and_rho.ALL.txt')
    if not osp.exists(path):
        print("++ WARNING: GS Kappa/Rho file missing: %s" % path)
        continue
    gs = pd.read_csv(path)
    kappa_rho_df.loc[(sbj,ses),'kappa (GS)'] = gs['kappa'][0]
    kappa_rho_df.loc[(sbj,ses),'rho (GS)'] = gs['rho'][0]
    if gs['kappa'][0] > gs['rho'][0]:
        kappa_rho_df.loc[(sbj,ses),'kappa_rho_color'] = 'lightgreen'
    else:
        kappa_rho_df.loc[(sbj,ses),'kappa_rho_color'] = 'red'
kappa_rho_df = kappa_rho_df.infer_objects()
kappa_rho_df.to_csv(f'./summary_files/{DATASET}_gs_kappa_rho.csv')

