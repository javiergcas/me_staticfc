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
# At this point data has been pre-processed with an afni_proc, which has performed the following operations:
#
# * Despiking, time shift correction, motion correction, registration to MNI space, masking, OC, tedana (fastica and robustica), scale and regress
# * The regression step, which takes as input ```pb05.$subj.r*.scale+tlrc.HEAD```, will include the following regressors:
#   * motion parameters and their first derivative
#   * compCorr physiological regressors
#   * fastANATICOR regressors **(NOT USED)**
#   * Bandpass filtering 0.01 - 0.2 Hz
#   * legengre polynomials up to 5th degree
#   * components marked as non-bold by tedana
#
#   > **NOTE**: If we want to include fanaticor, we would have to add ```Local_FSWe_rall``` as a regressor in all other pipelines.
#
# Program ```afni_proc``` is meant to run a traditional me analysis, meaning it will combine echoes and denoised the combined echo data. In this project, becuase we are interested on estimating FC across the echoes, we need to denoise the echoes separately. The N03a-x notebooks help us do this.
#
# In particular we need echo denoised data following 4 different pipelines. 
# ***
# * **Basic pipeline**: relies on afni_proc outputs up to the mask step. Then it does:
#     * computes scaled version of ```pb03.${SBJ}.r01.${EC}.volreg+tlrc``` (e.g., pre-tedana)
#     * regression: motion + 1rt der, compCorr, banspass, legendre (up to 5th)
#     * **output 4D**: ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic```
#     * **output TS**: ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}_000.netts```
#     * **output FC**: ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}_000.netcc```
#
#     > **NOTE**: This pipeline is implemented in this notebook
# ***
#
# * **Global Signal Regression pipeline**: relies on afni_proc outputs up to the mask step. Then it does:
#     * computes scaled version of ```pb03.${SBJ}.r01.${EC}.volreg+tlrc```
#     * extracts GS from that scaled version in two ways (as is & after detrending with 5th order legendre polynomials)
#     * regression: motion + 1rt der, compCorr, banspass, legendre (up to 5th), and GS
#     * **output 4D**: ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GS```
#     * **output TS**: ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GS.${ATLAS_NAME}_000.netts```
#     * **output FC**: ```errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GS.${ATLAS_NAME}_000.netcc```
#
#     > **NOTE**: This pipeline is implemented in this notebook
# ***
#
# * **TEDANA pipepline (fastica)**:
#
#     * Same as **Basic** pipeline, but also includes as extra nuisance regressors components marked as "bad" by tedana (run using the fastica pipeline)
#
# ***
#
# * **TEDANA pipepline (robustica)**:
#
#     * Same as **Basic** pipeline, but also includes as extra nuisance regressors components marked as "bad" by tedana (run using the robustica pipeline)
#       
# ***

import pandas as pd
import numpy as np
from glob import glob
import os.path as osp
import os
import subprocess
import datetime
from tqdm import tqdm
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, PRJ_DIR, CODE_DIR
from utils.basics import get_dataset_index
from sfim_lib.io.afni import load_netcc
from sfim_lib.plotting.fc_matrices import hvplot_fc

import getpass
username = getpass.getuser()
print(username)

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

DATASET='evaluation'

ATLAS_NAME = f'Power264-{DATASET}'

ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

# # 1. Load Dataset Information

ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())

# # 2. Create Swarm Script to Extract ROI TS from Basic denoised data

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N03a_Basic_and_GSR_pipeline_and_ROIextract.{ATLAS_NAME}.SWARM.sh')
print(script_path)

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N03a_Basic_and_GSR_pipeline_and_ROIextract.{ATLAS_NAME}.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 2 --time 02:00:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for sbj,ses in list(dataset_info_df.index):
        atlas_path  = f'{ATLASES_DIR}/{ATLAS_NAME}/{ATLAS_NAME}.nii.gz'
        for NORDIC in ['on','off']:
            the_file.write(f'export SBJ={sbj} SES={ses} NORDIC={NORDIC} ATLAS_NAME={ATLAS_NAME} ATLAS_PATH={atlas_path} ATLASES_DIR={ATLASES_DIR}; sh  {CODE_DIR}/bash/S09_Basic_and_GSR_pipeline_and_ROIextract.sh \n')
the_file.close()     

script_path

# ```bash
# # cd /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N03a_Basic_and_GSR_pipeline_and_ROIextract.Power264-evaluation.SWARM.sh -g 16 -t 8 -b 10 --time 02:00:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N03a_Basic_and_GSR_pipeline_and_ROIextract.Power264-evaluation.log --partition quick,norm --module afni

# # 3. Check all expected datasets were processed

for sbj,ses in list(dataset_info_df.index):
    netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.e02.volreg.scale.tproject_ALL_Basic.{ATLAS_NAME}_000.netcc')
    if not osp.exists(netcc_path):
        print('++ WARNING: %s is missing'% netcc_path)
    netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-on',f'errts.{sbj}.r01.e02.volreg.scale.tproject_ALL_GS.{ATLAS_NAME}_000.netcc')
    if not osp.exists(netcc_path):
        print('++ WARNING: %s is missing'% netcc_path)

# # 4. Load Motion information per scan

FINAL_N_ACQS = 201
mot_df = pd.DataFrame(index=np.arange(FINAL_N_ACQS),columns=list(dataset_info_df.index))
for sbj,ses in tqdm(list(dataset_info_df.index)):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    mot_df[(sbj,ses)] = np.loadtxt(mot_path)

# # 4. Explore the FC matrices just computed
#
# At this point, only within TE matrices (e.g., TE1_to_TE1, TE2_to_TE2, etc.) are available. No FC across echoes exists yet.

import xarray as xr
import panel as pn

roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
print(roi_info_path)
print(roi_info_df.shape)

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

fcs = xr.DataArray(dims=['censor','scan','NORDIC','roi_x','roi_y'], 
                   coords={'censor':['ALL_Basic','ALL_GS','KILL_Basic','KILL_GS'],
                           'scan':['.'.join([sbj,ses]) for sbj,ses in dataset_info_df.index],
                           'NORDIC':['off','on'],
                           'roi_x':list(roi_info_df['ROI_Name']),'roi_y':list(roi_info_df['ROI_Name'])})

for sbj,ses in tqdm(list(dataset_info_df.index)):
    for scenario in ['ALL_Basic','ALL_GS','KILL_Basic','KILL_GS'] :
        for NORDIC in ['off','on']:
            netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-{NORDIC}',f'errts.{sbj}.r01.e02.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netcc') 
            netcc = load_netcc(netcc_path)
            fcs.loc[scenario,'.'.join([sbj,ses]),NORDIC,:,:] = netcc.values

scan_select   = pn.widgets.Select(name='scan', options=list(fcs.coords['scan'].values))
NORDIC_select = pn.widgets.Select(name='NORDIC', options=['off','on'])
@pn.depends(scan_select, NORDIC_select)
def plot_fc(scan,NORDIC):
    aux = fcs.sel(censor='ALL_Basic',scan=scan, NORDIC=NORDIC).values
    aux = pd.DataFrame(aux,index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index, 
                           columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
    plot_all = hvplot_fc(aux,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=True, colorbar_position='left', net_cmap=power264_nw_cmap, cbar_title='FC-R (All Acquisitions)')
    aux = fcs.sel(censor='ALL_GS',scan=scan, NORDIC=NORDIC).values
    aux = pd.DataFrame(aux,index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index, 
                           columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
    plot_kill = hvplot_fc(aux,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=True, colorbar_position='left', net_cmap=power264_nw_cmap, cbar_title='FC-R (All acquisitions + GSasis)')
    
    return plot_all + plot_kill


@pn.depends(scan_select)
def plot_mot(scan):
    sbj,ses = scan.split('.')
    aux_df = pd.DataFrame(mot_df[(sbj,ses)].values,columns=['Motion [enorm]'])
    aux_df.index.name = 'TR'
    aux_df.name = 'Motion'
    return aux_df.hvplot(width=1000,c='k')


dashboard = pn.Row(pn.Column(scan_select,NORDIC_select),pn.Column(plot_fc,plot_mot)).show(port=port_tunnel)

dashboard.stop()
