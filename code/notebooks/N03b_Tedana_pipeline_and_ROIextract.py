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

# # Description
#
# This notebook implements the extra steps (beyond ```afni_proc```) needed to complete the per-echo **TEDANA pipeline** and **TEDANA+GSR pipeline**, which includes the following operations:
# ***
# * **TEDANA Pipeline**:
#     *  correct space in header for tedana outputs containing the per-echo denoised timeseries --> ```pb06.${SBJ}.r01.e01.meica_dn```
#     *  created scaled version --> ```pb07.${SBJ}.r01.${EC}.meica_dn.scale```
#     *  regression including same as basic (motion + 1rt der, compCorr, banspass, legendre (up to 5th)) on file where bad components have been regressed --> ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Tedana```
#
# > **NOTE**: final denoised timeseries include ```volreg``` instead of ```meica_dn``` to facilitate loading files later). Pipelines are indexed only by the final part of the file name
# ***
# * **TEDANA+GSR Pipeline**:
#     *  compute GS using ```pb07.${SBJ}.r01.${EC}.meica_dn.scale``` as input --> ```pb07.${SBJ}.r01.${EC}.meica_dn.scale.GSasis.1D```
#     *  regression including same as GSR (motion + 1rt der, compCorr, banspass, legendre (up to 5th), GS) on file where bad components have been regressed --> ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_TedanaGSasis```
# ***
# > **NOTE:** As of Jan,29th 2025 I have not implemented the additional pipelines (e.g., 3dPFM or Rapidtide). These should be implemented as N03c, N03d, etc.

import pandas as pd
import xarray as xr
import numpy as np
from glob import glob
import os.path as osp
import panel as pn
import subprocess
import datetime
import os
from sfim_lib.io.afni import load_netcc
from sfim_lib.plotting.fc_matrices import hvplot_fc
from tqdm import tqdm
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, ATLAS_NAME, PRJ_DIR, CODE_DIR


import getpass
username = getpass.getuser()
print(username)

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

# # 1. Load Dataset Information

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])

# # 2. Create Swarm Script to Extract ROI TS from fully denoised data

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N03b_Tedana_pipeline_and_ROIextract.{ATLAS_NAME}.SWARM.sh')
print(script_path)

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N03b_Tedana_pipeline_and_ROIextract.{ATLAS_NAME}.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 10 --time 00:20:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for sbj,ses in list(dataset_info_df.index):
        atlas_path  = f'{ATLASES_DIR}/{ATLAS_NAME}/{ATLAS_NAME}.nii.gz' 
        the_file.write(f'export SBJ={sbj} SES={ses} ATLAS_NAME={ATLAS_NAME} ATLAS_PATH={atlas_path} ATLASES_DIR={ATLASES_DIR}; sh  {CODE_DIR}/bash/N03b_Tedana_pipeline_and_ROIextract.sh \n')
the_file.close()     

script_path

# ```bash
# # cd /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N03b_Tedana_pipeline_and_ROIextract.Power264.SWARM.sh -g 16 -t 8 -b 10 --time 00:20:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N03b_Tedana_pipeline_and_ROIextract.Power264.log --partition quick,norm --module afni
# ```
#
# ***

# # 3. Check all expected datasets were processed

# %%time
for sbj,ses in list(dataset_info_df.index):
    for scenario in ['ALL_Tedana','KILL_Tedana','ZERO_Tedana','NTRP_Tedana','ALL_TedanaGSasis','KILL_TedanaGSasis','ZERO_TedanaGSasis','NTRP_TedanaGSasis']:
        for e in ['e01','e02','e03']:
            netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
            if not osp.exists(netcc_path):
                print('++ WARNING: %s is missing'% netcc_path)

# ***
#
# # 4. Create a small dashboard to explore FC matrices for different pipelines

import xarray as xr
import panel as pn

roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

# +
scenarios = [('Basic','ALL','e'+str(i+1).zfill(2)) for i in np.arange(3)] + [('GSasis','ALL','e'+str(i+1).zfill(2)) for i in np.arange(3)] + [('Tedana','ALL','e'+str(i+1).zfill(2)) for i in np.arange(3)]

fcs = {scenario:xr.DataArray(dims=['scan','roi_x','roi_y'], coords={'scan':['.'.join([sbj,ses]) for sbj,ses in dataset_info_df.index],
                                                                           'roi_x':list(roi_info_df['ROI_Name']),
                                                                           'roi_y':list(roi_info_df['ROI_Name'])}) for scenario in scenarios}
# -

# ### Load FC matrices

for sbj,ses in tqdm(list(dataset_info_df.index)):
    for pipeline,censor_mode,te in scenarios:
        netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{te}.volreg.scale.tproject_{censor_mode}_{pipeline}.{ATLAS_NAME}_000.netcc')
        netcc = load_netcc(netcc_path)
        fcs[(pipeline,censor_mode,te)].loc['.'.join([sbj,ses]),:,:] = netcc.values

# ### Load Motion Timeseries

mot_df = pd.DataFrame(index=np.arange(201),columns=list(dataset_info_df.index))
for sbj,ses in tqdm(list(dataset_info_df.index)):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'motion_{sbj}_enorm.1D')
    mot_df[(sbj,ses)] = np.loadtxt(mot_path)

# ### Dashboard Elements and Functions

scan_select = pn.widgets.Select(name='scan', options=list(dataset_info_df.index))
@pn.depends(scan_select)
def plot_fc(scan):
    sbj = scan[0]
    ses = scan[1]
    scan_index = '.'.join([sbj,ses])
    plots = pn.layout.GridBox(ncols=3)
    for scenario in scenarios:
        aux = fcs[scenario].sel(scan=scan_index).values
        aux = pd.DataFrame(aux,index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index,
                               columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
        aux_plot = hvplot_fc(aux,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', net_cmap=power264_nw_cmap, cbar_title='%s-%s-%s' % scenario)
        plots.append(aux_plot)
    return pn.Row(plots)


@pn.depends(scan_select)
def plot_mot(scan):
    aux_df = pd.DataFrame(mot_df[scan].values,columns=['Motion [enorm]'])
    aux_df.index.name = 'TR'
    aux_df.name = 'Motion'
    return aux_df.hvplot()


# ## Start Dashboard

dashboard = pn.Row(scan_select,pn.Column(plot_fc,plot_mot)).show(port=port_tunnel)

mot_df.mean().sort_values()
