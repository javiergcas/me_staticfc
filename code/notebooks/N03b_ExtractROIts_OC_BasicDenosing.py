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
# This notebook extract ROI timeseries and computes FC on an optimally-combined + Basic denoised version of the data.
#
# It also generated a little dashboard to look at those in comparison to the E02 + Basic denoised data.

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
from nilearn.connectome import sym_matrix_to_vec
import holoviews as hv

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

# # 2. Create Swarm Script to Extract ROI TS OC + Basic Denoised Data

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N03b_extract_ROI_ts.{ATLAS_NAME}.OC_BasicDenosing.sh')
print(script_path)

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N03b_extract_ROI_ts.{ATLAS_NAME}.OC_BasicDenosing.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 10 --time 00:20:00 --logdir {log_path} --partition quick,norm --module afni --sbatch \"--export AFNI_COMPRESSOR=GZIP\"\n')
    the_file.write('\n')
    for sbj,ses in list(dataset_info_df.index):
        atlas_path  = f'{ATLASES_DIR}/{ATLAS_NAME}/{ATLAS_NAME}.nii.gz' 
        inset_path  = f'errts.{sbj}.tproject+tlrc'
        the_file.write(f'export SBJ={sbj} SES={ses} ATLAS_NAME={ATLAS_NAME} ATLAS_PATH={atlas_path} ATLASES_DIR={ATLASES_DIR}; sh  {CODE_DIR}/bash/N03b_ExtractROIts_OC_Basic_Denoising.sh \n')
the_file.close()     

# # 3. Instructions to run code via Swarm

script_path

# ```bash
# # cd /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N03b_extract_ROI_ts.Power264.OC_BasicDenosing.sh -g 16 -t 8 -b 10 --time 00:20:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N03b_extract_ROI_ts.Power264.OC_BasicDenosing.log --partition quick,norm --module afni

# ***
# # 4. Check all expected datasets were processed

for sbj,ses in list(dataset_info_df.index):
    for scenario in ['ALL','ALL_GSasis','ALL_GSdt5','KILL','KILL_GSasis','KILL_GSdt5']:
        netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.OC.tproject_{scenario}.{ATLAS_NAME}_000.netts')
        if not osp.exists(netcc_path):
            print('++ WARNING: %s is missing'% netcc_path)

# ***
#
# # 5. Plot FC matrices
#
# ## 5.1. Load ROI Information

roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

# ## 5.2. Load FC matrices for Basic denoising (E02 and OC) 

# +
fcs = {}
for scenario in ['E02-basic','OC-basic','OC-basic-GSasis','OC-basic-GSdt5']:
    fcs[scenario] = xr.DataArray(dims=['scan','roi_x','roi_y'], coords={'scan':['.'.join([sbj,ses]) for sbj,ses in dataset_info_df.index],'roi_x':list(roi_info_df['ROI_Name']),'roi_y':list(roi_info_df['ROI_Name'])})
    
#fcs = {'E02-basic':   xr.DataArray(dims=['scan','roi_x','roi_y'], coords={'scan':['.'.join([sbj,ses]) for sbj,ses in dataset_info_df.index],'roi_x':list(roi_info_df['ROI_Name']),'roi_y':list(roi_info_df['ROI_Name'])}),
#       'OC-basic':xr.DataArray(dims=['scan','roi_x','roi_y'], coords={'scan':['.'.join([sbj,ses]) for sbj,ses in #dataset_info_df.index],'roi_x':list(roi_info_df['ROI_Name']),'roi_y':list(roi_info_df['ROI_Name'])})}
# -

# %%time
for sbj,ses in tqdm(list(dataset_info_df.index)):
    netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.e02.volreg.scale.tproject_ALL.{ATLAS_NAME}_000.netcc')#f'pb03.{sbj}.e02.volreg.{ATLAS_NAME}_000.netcc')
    netcc = load_netcc(netcc_path)
    fcs['E02-basic'].loc['.'.join([sbj,ses]),:,:] = netcc.values

# %%time
for sbj,ses in tqdm(list(dataset_info_df.index)):
    netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.OC.tproject_ALL.{ATLAS_NAME}_000.netcc')# f'errts.{sbj}.tproject.{ATLAS_NAME}_000.netcc')
    netcc = load_netcc(netcc_path)
    fcs['OC-basic'].loc['.'.join([sbj,ses]),:,:] = netcc.values

# %%time
for sbj,ses in tqdm(list(dataset_info_df.index)):
    netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.OC.tproject_ALL_GSasis.{ATLAS_NAME}_000.netcc')# f'errts.{sbj}.tproject.{ATLAS_NAME}_000.netcc')
    netcc = load_netcc(netcc_path)
    fcs['OC-basic-GSasis'].loc['.'.join([sbj,ses]),:,:] = netcc.values

# %%time
for sbj,ses in tqdm(list(dataset_info_df.index)):
    netcc_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.OC.tproject_ALL_GSdt5.{ATLAS_NAME}_000.netcc')# f'errts.{sbj}.tproject.{ATLAS_NAME}_000.netcc')
    netcc = load_netcc(netcc_path)
    fcs['OC-basic-GSdt5'].loc['.'.join([sbj,ses]),:,:] = netcc.values

# ## 5.3 Load Motion information for reference

mot_df = pd.DataFrame(index=np.arange(201),columns=list(dataset_info_df.index))
for sbj,ses in tqdm(list(dataset_info_df.index)):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'motion_{sbj}_enorm.1D')
    mot_df[(sbj,ses)] = np.loadtxt(mot_path)

# ## 5.4. Construct Dashboard 

scan_select = pn.widgets.Select(name='scan', options=list(fcs['E02-basic'].coords['scan'].values))
@pn.depends(scan_select)
def plot_fc(scan):
    aux = fcs['OC-basic'].sel(scan=scan).values
    aux_vec_e02 = sym_matrix_to_vec(aux, discard_diagonal=True)
    aux = pd.DataFrame(aux,index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index, columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
    aux_plot_volreg = hvplot_fc(aux,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', net_cmap=power264_nw_cmap, cbar_title='E02-basic')

    aux = fcs['OC-basic-GSasis'].sel(scan=scan).values
    aux_vec_oc = sym_matrix_to_vec(aux, discard_diagonal=True)
    aux = pd.DataFrame(aux,index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index, columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
    aux_plot_meica = hvplot_fc(aux,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', net_cmap=power264_nw_cmap, cbar_title='OC-basic')

    df = pd.DataFrame([aux_vec_e02,aux_vec_oc], index=['E02-Basic','OC-Basic']).T
    scat_plot = df.hvplot.scatter(x='E02-Basic',y='OC-Basic', aspect='square', datashade=True) * hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2)
    return pn.Row(aux_plot_volreg, aux_plot_meica, scat_plot)


@pn.depends(scan_select)
def plot_mot(scan):
    sbj,ses = scan.split('.')
    aux_df = pd.DataFrame(mot_df[(sbj,ses)].values,columns=['Motion [enorm]'])
    aux_df.index.name = 'TR'
    aux_df.name = 'Motion'
    return aux_df.hvplot(width=1500,c='k')


dashboard = pn.Row(scan_select,pn.Column(plot_fc,plot_mot)).show(port=port_tunnel)

dashboard.stop()

mot_df.sum().sort_values()
