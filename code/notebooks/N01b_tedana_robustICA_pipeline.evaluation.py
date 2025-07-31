# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: BOLD WAVES 2024a
#     language: python
#     name: bold_waves_2024a
# ---

# # Description
#
# This dataset will create the swarm jobs to run tedana with robustICA both with NORDIC on and off on the **evaluation** dataset

# +
import os
import datetime

import pandas as pd
import os.path as osp
from utils.basics import PRJ_DIR, CODE_DIR, PRCS_DATA_DIR
# -

import getpass
username = getpass.getuser()
print(username)

# # 1. Load Dataset Information

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])

# # 2. Create Swarm Script to Extract ROI TS from fully denoised data

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'S08_tedana-robustICA_evaluation.SWARM.sh')
print(script_path)

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'S08_tedana-robustICA_evaluation.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 64 -t 8 -b 4 --time 00:50:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for sbj,ses in list(dataset_info_df.index):
        the_file.write(f'export SBJ={sbj} SES={ses} DATASET=evaluation NORDIC=on; sh  {CODE_DIR}/bash/S08_tedana-robustICA.sh \n')
        the_file.write(f'export SBJ={sbj} SES={ses} DATASET=evaluation NORDIC=off; sh  {CODE_DIR}/bash/S08_tedana-robustICA.sh \n')
the_file.close()     

script_path

# ```bash
# # cd /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/S08_tedana-robustICA_evaluation.SWARM.sh -g 64 -t 8 -b 10 --time 00:20:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/S08_tedana-robustICA_evaluation.log --partition quick,norm --module afni
# ```
#
# ***

# # 3. Check all expected datasets were processed

# %%time
missing_cases = []
for sbj,ses in list(dataset_info_df.index):
    for NORDIC in ['on','off']:
        for e in ['e01','e02','e03']:
            path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}','tedana_fastica','tedana_report.html')
            if not osp.exists(path):
                print('++ WARNING: %s is missing'% path)
                missing_cases.append((sbj,ses))

' '.join([f'-e "export SBJ={sbj} SES={ses}"' for sbj, ses in list(set(missing_cases))])

print(len(missing_cases))


