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

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N03c_ExtractROIts_TEDANA.{ATLAS_NAME}.sh')
print(script_path)

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N03c_ExtractROIts_TEDANA.{ATLAS_NAME}.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 10 --time 00:20:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for sbj,ses in list(dataset_info_df.index):
        atlas_path  = f'{ATLASES_DIR}/{ATLAS_NAME}/{ATLAS_NAME}.nii.gz' 
        the_file.write(f'export SBJ={sbj} SES={ses} ATLAS_NAME={ATLAS_NAME} ATLAS_PATH={atlas_path} ATLASES_DIR={ATLASES_DIR}; sh  {CODE_DIR}/bash/N03c_ExtractROIts_TEDANA.sh \n')
the_file.close()     

script_path

# ```bash
# # cd /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N03c_ExtractROIts_TEDANA.Power264.sh -g 16 -t 8 -b 10 --time 00:20:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N03d_Post_AfniProc.Power264.log --partition quick,norm --module afni
# ```

# # 3. Check all expected datasets were processed

# %%time
for sbj,ses in list(dataset_info_df.index):
    for scenario in ['ALL','KILL','ZERO','NTRP','ALL_GSasis','KILL_GSasis','ZERO_GSasis','NTRP_GSasis','ALL_GSasis','KILL_GSasis','ZERO_GSasis','NTRP_GSasis']:
        for e in ['e01','e02','e03']:
            netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e}.meica_dn.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
            if not osp.exists(netcc_path):
                print('++ WARNING: %s is missing'% netcc_path)

# ***

roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df

scenarios = ['E01-volreg','E02-volreg','E03-volreg','E01-MEICA','E02-MEICA','E03-MEICA']
fcs = {scenario:xr.DataArray(dims=['scan','roi_x','roi_y'], coords={'scan':['.'.join([sbj,ses]) for sbj,ses in dataset_info_df.index],
                                                                           'roi_x':list(roi_info_df['ROI_Name']),
                                                                           'roi_y':list(roi_info_df['ROI_Name'])}) for scenario in scenarios}

for sbj,ses in tqdm(list(dataset_info_df.index)):
    for scenario in scenarios:
        e = (scenario.split('-')[0]).lower()
        if 'volreg' in scenario:
            netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.{e}.volreg.{ATLAS_NAME}_000.netcc')
        if 'MEICA' in scenario:
            netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e}-MEICA.tproject.{ATLAS_NAME}_000.netcc')
        netcc = load_netcc(netcc_path)
        fcs[scenario].loc['.'.join([sbj,ses]),:,:] = netcc.values

mot_df = pd.DataFrame(index=np.arange(201),columns=list(dataset_info_df.index))
for sbj,ses in tqdm(list(dataset_info_df.index)):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'motion_{sbj}_enorm.1D')
    mot_df[(sbj,ses)] = np.loadtxt(mot_path)

scan_select = pn.widgets.Select(name='scan', options=list(fcs['E02-volreg'].coords['scan'].values))
@pn.depends(scan_select)
def plot_fc(scan):
    plots = pn.layout.GridBox(ncols=3)
    for scenario in scenarios:
        aux = fcs[scenario].sel(scan=scan).values
        aux = pd.DataFrame(aux,index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index,
                               columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
        aux_plot = hvplot_fc(aux,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', net_cmap=nw_color_dict)
        plots.append(aux_plot)
    return pn.Row(plots)


@pn.depends(scan_select)
def plot_mot(scan):
    sbj,ses = scan.split('.')
    aux_df = pd.DataFrame(mot_df[(sbj,ses)].values,columns=['Motion [enorm]'])
    aux_df.index.name = 'TR'
    aux_df.name = 'Motion'
    return aux_df.hvplot()


dashboard = pn.Row(scan_select,pn.Column(plot_fc,plot_mot)).show(port=port_tunnel)

dashboard.stop()

import holoviews as hv

hv.help(pn.layout.FlexBox)


