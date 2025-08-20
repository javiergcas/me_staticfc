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
#     display_name: Generic Kernel (2025a)
#     language: python
#     name: generic_2025a
# ---

# # Description: Compute Kappa and Rho for Global Signal
#
# In this notebook we estimate the kappa and the rho of the global signal

import os.path as osp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from utils.basics import PRJ_DIR, PRCS_DATA_DIR, SPRENG_DOWNLOAD_DIR, CODE_DIR

import getpass
username = getpass.getuser()
print(username)

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])

# # 1. Create batch jobs for computing kappa and rho for all scans
#
# We will use the ```generate_metrics``` function within ```tedana``` to compute the kappa and the rho
#
# ## 1.1. Create the path to the swarm file

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N10_Compute_GS_kappa_and_rho.SWARM.sh')
print(script_path)

# ## 1.2. Create a folder for log files

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N10_Compute_GS_kappa_and_rho.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

# ## 1.3. Write the swarm file
#
# This file will contain one line per scan and calls the program ```GS_kappa_and_rho.py```.

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for sbj,ses in tqdm(dataset_info_df.index):
        the_file.write(f'export SBJ={sbj} RUN={ses}; cd {CODE_DIR}/python/; sh {CODE_DIR}/python/GS_kappa_and_rho.sh \n')
the_file.close()     

# ***
# # 2. Check all jobs finalized correctly
#
# As we do that, we will also create a .csv file ```gs_kappa_rho.csv``` with the estimated kappa and rho for all scans. 
#
# > **NOTE:** If you see any warnings, that means that the associated batch job did not finish correctly. Try again.
#
# We will be able to load that in other notebooks.

kappa_rho_df = pd.DataFrame(index=dataset_info_df.index,columns=['kappa (GS)','rho (GS)','kappa_rho_color'])
for sbj,ses in tqdm(dataset_info_df.index):
    path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'{sbj}_{ses}_GS_kappa_and_rho.txt')
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

kappa_rho_df.to_csv('./cache/gs_kappa_rho.csv')

# ***
# # Extra Code: used to test the intial version of ```GS_kappa_and_rho.sh```

from tedana.metrics.collect import generate_metrics
from tedana.io import OutputGenerator
import numpy as np
import pandas as pd
import nibabel as nib
import os.path as osp
from utils.basics import PRCS_DATA_DIR, TES_MSEC, PRJ_DIR
import hvplot.pandas
from scipy.stats import zscore
from tqdm import tqdm

tes = list(TES_MSEC['Spreng_Scanner1'].values())
ne  = len(tes)
sbj,ses='sub-01','ses-1'

# Load the adaptive mask
mask_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','tedana_r01','adaptive_mask.nii.gz')
mask_img  = nib.load(mask_path)
mask_data = mask_img.get_fdata()
nx,ny,nz  = mask_data.shape
mask_vec  = mask_data.reshape(nx*ny*nz,).astype(int)

# Extract number of acquisitions from first echo
e1_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e01.volreg+tlrc.HEAD')
e1_img  = nib.load(e1_path)
e1_data = e1_img.get_fdata()
_,_,_,nt = e1_data.shape

data_cat = np.zeros((nx*ny*nz,ne,nt))
for e,ee in enumerate(tqdm(list(TES_MSEC['Spreng_Scanner1'].keys()))):
    path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.{ee}.volreg+tlrc.HEAD')
    img  = nib.load(path)
    data = img.get_fdata()
    data_cat[:,e,:] = data.reshape(nx*ny*nz,nt)

# Load the Global Signal
gs_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e02.volreg.scale.GSasis.1D')
gs      = zscore(np.loadtxt(gs_path)).reshape(nt,1)
gs.shape

# Load the Optimally combined data
oc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','tedana_r01','ts_OC.nii.gz')
oc_img  = nib.load(oc_path)
oc_data = oc_img.get_fdata()
data_optcom = oc_data.reshape(nx*ny*nz,nt)

# Create Output Generator Object that will write nothing
io_generator = OutputGenerator(reference_img=mask_img,
        convention='orig',
        out_dir=osp.join(PRJ_DIR,'temp'),
        prefix=f'temp.{sbj}.{ses}',
        config="auto",
        overwrite=True,
        make_figures=False,
        verbose=False)

component_table, mixing = generate_metrics(data_cat=data_cat,
                 data_optcom=data_optcom, 
                 mixing=gs,
                 adaptive_mask=mask_vec,
                 tes=tes,
                 io_generator=io_generator,
                 label='GS',
                 metrics=['kappa','rho'])

component_table
