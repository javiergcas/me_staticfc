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

from nilearn.datasets import fetch_coords_power_2011
from utils.basics import ATLASES_DIR
import os
import os.path as osp
import pandas as pd

power_atlas_info = fetch_coords_power_2011(False)

power_atlas_info['rois'].head(20)

ATLAS_NAME='Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

if not osp.exists(ATLAS_DIR):
    os.makedirs(ATLAS_DIR)

roi_centers_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_coords.MNI.csv')
power_atlas_info['rois'][['x','y','z','roi']].to_csv(roi_centers_path, header=None, index=None)

roi_info_df = power_atlas_info['rois'].copy()
roi_info_df.columns = ['ROI_ID','pos_A','pos_R','pos_S']
roi_info_df['ROI_Name'] = ['ROI'+str(r).zfill(3) for r in roi_info_df['ROI_ID']]
roi_info_df = roi_info_df[['ROI_ID','ROI_Name','pos_A','pos_R','pos_S']]
print(roi_info_df.shape)
roi_info_df.head(5)

# +
color_map_dict={'White':'#ffffff','Cyan':'#E0FFFF','Orange':'#FFA500','Purple':'#800080',
                'Pink':'#FFC0CB','Red':'#ff0000','Gray':'#808080','Teal':'#008080','Brown':'#A52A2A',
                'Blue':'#0000ff','Yellow':'#FFFF00','Black':'#000000','Pale blue':'#ADD8E6','Green':'#00ff00'}
nw_color_dict = {'Uncertain':'#ffffff',
                 'Sensory/somatomotor Hand':'#E0FFFF',
                 'Sensory/somatomotor Mouth':'#FFA500',
                 'Cingulo-opercular Task Control':'#800080',
                 'Auditory':'#FFC0CB',
                 'Default mode':'#ff0000',
                 'Memory retrieval?':'#808080',
                 'Ventral attention':'#008080',
                 'Visual':'#0000ff',
                 'Fronto-parietal Task Control':'#FFFF00',
                 'Salience':'#000000',
                 'Subcortical':'#A52A2A',
                 'Cerebellar':'#ADD8E6',
                 'Dorsal attention':'#00ff00'}

power_atlas_addinfo_path = osp.join(ATLAS_DIR,'additional_files','Neuron_consensus_264.xlsx')
power_atlas_addinfo = pd.read_excel(power_atlas_addinfo_path, header=[0], skiprows=[1])

roi_info_df['Network']= power_atlas_addinfo['Suggested System']
roi_info_df['Hemisphere'] = ['LH' if a<=0 else 'RH' for a in roi_info_df['pos_R']]
roi_info_df['RGB'] = [color_map_dict[c] for c in power_atlas_addinfo['Unnamed: 34']]
roi_info_df.head(5)
# -

# ```bash
#     ml afni
#     cd /data/SFIMJGC_HCP7T/BCBL2024/atlases/Power264
#     3dUndump -overwrite \
#              -prefix Power264.nii.gz \
#              -master ../../prcs_data/sub-01/D02_Preproc_fMRI_ses-1/errts.sub-01.fanaticor+tlrc.HEAD \
#              -xyz \
#              -srad 5 \
#              -xyz Power264.roi_coords.MNI.csv
# ```

# ***

import pandas as pd
from glob import glob
import os.path as osp
import subprocess
import datetime
import os
from utils.basics import PRCS_DATA_DIR, PRJ_DIR, CODE_DIR
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

import getpass
username = getpass.getuser()
print(username)

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])

# ***

script_path = osp.join(PRJ_DIR,f'swarm.{username}','N02b_check_sample_FOV_vs_atlas.Power264.swarm.sh')
print(script_path)

log_path = osp.join(PRJ_DIR,f'logs.{username}','N02b_check_sample_FOV_vs_atlas.Power264.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 5 --time 00:20:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for sbj,ses in list(dataset_info_df.index):
        the_file.write(f'cd {PRCS_DATA_DIR}/{sbj}/D02_Preproc_fMRI_{ses}; 3dcalc -overwrite -a tedana_r01/adaptive_mask.nii.gz -expr "step(a)" -prefix mask_tedana_at_least_one_echo.nii.gz; 3dcalc -overwrite -a tedana_r01/adaptive_mask.nii.gz -expr "equals(a,3)" -prefix mask_tedana_allechoes.nii.gz; 3drefit -space MNI mask_tedana_at_least_one_echo.nii.gz; 3drefit -space MNI mask_tedana_allechoes.nii.gz; 3dNetCorr -overwrite -in_rois {ATLASES_DIR}/{ATLAS_NAME}/{ATLAS_NAME}.nii.gz -output_mask_nonnull -inset pb04.{sbj}.r01.combine+tlrc.HEAD -prefix rm.{sbj}.combine.{ATLAS_NAME}.FOVcheck \n')
the_file.close()     

script_path

# You need to submit this as a batch job
# ```bash
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N02b_check_sample_FOV_vs_atlas.Power264.swarm.sh -g 16 -t 8 -b 5 --time 00:20:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N02b_check_sample_FOV_vs_atlas.Power264.log --partition quick,norm --module afni
# ```

for sbj,ses in list(dataset_info_df.index):
    expected_output_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'rm.{sbj}.combine.{ATLAS_NAME}.FOVcheck_mask_nnull+tlrc.HEAD')
    if not osp.exists(expected_output_path):
        print('++ WARNING: %s is missing' % expected_output_path)

# ***

bad_roi_list = []
for sbj,ses in list(dataset_info_df.index):
    roidat_path       = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'rm.{sbj}.combine.{ATLAS_NAME}.FOVcheck_000.roidat')
    roidat_df         = pd.read_csv(roidat_path,sep=' ', skipinitialspace=True, header=0)
    correct_columns   = roidat_df.columns.drop(['#'])
    roidat_df         = roidat_df.drop(['ROI_label'],axis=1)
    roidat_df.columns = correct_columns
    roidat_df         = roidat_df.drop(['#.1'],axis=1)
    bad_rois          = roidat_df[(roidat_df['frac']<=0.05) | (roidat_df['N_nonnull']<10)][['ROI','ROI_label']]
    if bad_rois.shape[0] > 0:
        print('++ INFO: %s/%s --> Number of Bad Rois: %d' % (sbj,ses,bad_rois.shape[0]), end=' | ')
    for i,br in bad_rois.iterrows():
        bad_roi_list.append((br['ROI'],br['ROI_label']))

bad_roi_list = list(set(bad_roi_list))

print(bad_roi_list)

print('++ INFO: Number of ROIs to remove = %d ROIs' % len(bad_roi_list))

roi_info_df = roi_info_df.drop([i for i,_ in bad_roi_list])
roi_info_df = roi_info_df.reset_index(drop=True)
roi_info_df['ROI_ID'] = roi_info_df.index
roi_info_df['ROI_Name'] = ['ROI'+str(r).zfill(3) for r in roi_info_df['ROI_ID']]
print(roi_info_df.shape)
roi_info_df.to_csv(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv'), index=False)

roi_info_df.reset_index(drop=True)

# ***

bad_rois_minus = '-'.join([str(r)+'*equals(a,'+str(r)+')' for r,rs in bad_roi_list])
bad_rois_plus  = '+'.join([str(r)+'*equals(a,'+str(r)+')' for r,rs in bad_roi_list])
print(bad_rois_minus)
print(bad_rois_plus)

command=f"""module load afni; \
           cd {ATLAS_DIR}; \
           3dcalc -overwrite \
                  -a {ATLAS_NAME}.nii.gz \
                  -expr '{bad_rois_plus}' \
                  -prefix {ATLAS_NAME}.RemovedROIs.nii.gz; \
           3dcalc -overwrite \
                  -a      {ATLAS_NAME}.nii.gz \
                  -expr 'a-{bad_rois_minus}' \
                  -prefix rm.{ATLAS_NAME}.fov_restricted.nii.gz; \
                  """
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

from nilearn.plotting import plot_roi

plot_roi(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.RemovedROIs.nii.gz'),title='ROIs that will be removed from the ATLAS')

plot_roi(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.nii.gz'),title='Original ATLAS')

plot_roi(osp.join(ATLAS_DIR,f'rm.{ATLAS_NAME}.fov_restricted.nii.gz'),title='FOV-Restricted ATLAS')

# ***

command = f"""ml afni; \
             cd {ATLAS_DIR}; \
             3dRank -overwrite -prefix {ATLAS_NAME}.nii.gz -input rm.{ATLAS_NAME}.fov_restricted.nii.gz;"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ***
# # Prepare Files for BrainNetViewer
#
# The first file will be loaded as node
#
# The second file must be loaded in the Node tab within the Options window

nw_list = list(roi_info_df['Network'].unique())
Nw2ID={r:i+1 for i,r in enumerate(nw_list)}

roi_info_df['Node Size'] = 1
roi_info_df['Short ROI Name'] = roi_info_df['ROI_Name']
roi_info_df['Node Color'] = [Nw2ID[n] for n in roi_info_df['Network']]
aux = roi_info_df[['pos_A','pos_R','pos_S','Node Color','Node Size','Short ROI Name']]
aux.to_csv('../../../resources/BrainNetViewer/BrainNetViewer_Nodes.node', sep=' ', index=None, header=None)

from matplotlib.colors import hex2color
import numpy as np

c = []
for n in nw_list:
    c = c + [np.array(hex2color(roi_info_df.set_index('Network').loc[n,'RGB'].values[0]))]

c = np.array(c)

np.savetxt('../../../resources/BrainNetViewer/BrainNetViewer_Nodes_colors.txt',c, fmt='%0.4f', delimiter=',')


