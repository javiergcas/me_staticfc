#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# This notebook will create a version of the Powers 264 altas that only contains ROIs common to the FOV of all datasets part of the Gating / Non-Gating dataset. 
# 
# This is the dataset that we use in the first part of the paper to describe our quality metrics

# In[1]:


from nilearn.datasets import fetch_coords_power_2011
from utils.basics import ATLASES_DIR
import os
import os.path as osp
import pandas as pd


# In[2]:


sbj_list = ['MGSBJ01',  'MGSBJ02',  'MGSBJ03',  'MGSBJ04',  'MGSBJ05',  'MGSBJ06',  'MGSBJ07']
ses_list = ['constant_gated', 'cardiac_gated']


# # 1. Download ROI centers for Powers 264 altas using nilearn

# In[3]:


power_atlas_info = fetch_coords_power_2011(False)


# In[4]:


power_atlas_info['rois'].head(5)


# # 2. Create Folder for Dataset-specific version of Powers 264 atlas

# In[5]:


# Build atlas identifiers and resolved output directory path for this dataset.
ATLAS_NAME='Power264-discovery'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
print(ATLAS_DIR)


# In[6]:


# Create atlas directory and link auxiliary files expected by downstream code.
if not osp.exists(ATLAS_DIR):
    os.makedirs(ATLAS_DIR)
    os.symlink(osp.join(ATLASES_DIR,'Power264','additional_files'),osp.join(ATLAS_DIR,'additional_files'))


# # 3. Write ROI centroids to disk as csv file

# In[7]:


# Save ROI MNI coordinates to the atlas-specific CSV used by the pipeline.
roi_centers_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_coords.MNI.csv')
power_atlas_info['rois'][['x','y','z','roi']].to_csv(roi_centers_path, header=None, index=None)
print("++ INFO: ROI Coordinates saved to disk [%s]" % roi_centers_path)


# # 4. Add ROI Names and other info needed for plotting

# In[8]:


# Normalize ROI table schema used by downstream FC extraction/plotting utilities.
roi_info_df = power_atlas_info['rois'].copy()
roi_info_df.columns = ['ROI_ID','pos_A','pos_R','pos_S']
roi_info_df['ROI_Name'] = ['ROI'+str(r).zfill(3) for r in roi_info_df['ROI_ID']]
roi_info_df = roi_info_df[['ROI_ID','ROI_Name','pos_A','pos_R','pos_S']]
print(roi_info_df.shape)
roi_info_df.head(5)


# In[9]:


# Map network labels to colors and merge supplemental metadata from the Power atlas sheet.
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


# # 5. Create first version (still need to check for FOV) as NIFTI file

# ```bash
#     ml afni
#     cd /data/SFIMJGC_HCP7T/BCBL2024/atlases/Power264-discovery
#     3dUndump -overwrite \
#              -prefix Power264-discovery.nii.gz \
#              -master ../../prcs_data/MGSBJ01/D03_Preproc_cardiac_gated_NORDIC-off/errts.MGSBJ01.fanaticor+tlrc.HEAD \
#              -xyz \
#              -srad 5 \
#              -xyz Power264-discovery.roi_coords.MNI.csv
# ```

# # 6. Create FOV masks for each dataset

# In[10]:


# Utilities for FOV QC swarm generation and AFNI command execution.
from utils.basics import PRCS_DATA_DIR, PRJ_DIR
import datetime
import getpass
import subprocess
username = getpass.getuser()
print(username)


# Create path for swarm file

# In[11]:


script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N02b_check_sample_FOV_vs_atlas.{ATLAS_NAME}.swarm.sh')
print(script_path)


# Create folder for swarm log files

# In[12]:


log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N02b_check_sample_FOV_vs_atlas.{ATLAS_NAME}.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)


# Create swarm file (one line per scan)

# In[13]:


# Build SWARM jobs to compute atlas-vs-FOV overlap masks for every scan.
with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 16 -t 8 -b 5 --time 00:20:00 --logdir {log_path} --partition quick,norm --module afni\n')
    the_file.write('\n')
    for sbj in sbj_list:
        for ses in ses_list:
            the_file.write(f'cd {PRCS_DATA_DIR}/{sbj}/D03_Preproc_{ses}_NORDIC-off; 3dcalc -overwrite -a tedana_fastica/adaptive_mask.nii.gz -expr "step(a)" -prefix mask_tedana_at_least_one_echo.nii.gz; 3dcalc -overwrite -a tedana_fastica/adaptive_mask.nii.gz -expr "equals(a,3)" -prefix mask_tedana_allechoes.nii.gz; 3drefit -space MNI mask_tedana_at_least_one_echo.nii.gz; 3drefit -space MNI mask_tedana_allechoes.nii.gz; 3dNetCorr -overwrite -in_rois {ATLASES_DIR}/{ATLAS_NAME}/{ATLAS_NAME}.nii.gz -output_mask_nonnull -inset pb04.{sbj}.r01.combine+tlrc.HEAD -prefix rm.{sbj}.combine.{ATLAS_NAME}.FOVcheck \n')
the_file.close()     


# In[14]:


script_path


# For this dataset, because the number of scans is so small, it might be ok to just run it a console. That said, if you want to still parallelize, here is the swarm call
# 
# ```bash
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N02b_check_sample_FOV_vs_atlas.Power264-discovery.sh -g 16 -t 8 -b 5 --time 00:10:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N02b_check_sample_FOV_vs_atlas.Power264-discovery.log --partition quick,norm --module afni
# ```

# # 7. Ensure all necessary files were created by the swarm job

# In[15]:


# Check that the expected FOV-check outputs were produced for each scan.
for sbj in sbj_list:
    for ses in ses_list:
        expected_output_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'rm.{sbj}.combine.{ATLAS_NAME}.FOVcheck_mask_nnull+tlrc.HEAD')
        if not osp.exists(expected_output_path):
            print('++ WARNING: %s is missing' % expected_output_path)


# # 8. See which ROIs do not have at least 5% overlap with the imaging FOV of any subject

# In[16]:


# Aggregate ROIs with poor coverage across scans using frac/N_nonnull thresholds.
bad_roi_list = []
for sbj in sbj_list:
    for ses in ses_list:
        roidat_path       = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'rm.{sbj}.combine.{ATLAS_NAME}.FOVcheck_000.roidat')
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


# In[17]:


bad_roi_list = list(set(bad_roi_list))


# In[18]:


print(bad_roi_list)


# In[19]:


print('++ INFO: Number of ROIs to remove = %d ROIs' % len(bad_roi_list))


# We now remove the bad ROIs from the ROI dataframe

# In[20]:


# Drop bad ROIs, re-index ROI IDs/names, and persist the updated ROI info table.
roi_info_df = roi_info_df.drop([i for i,_ in bad_roi_list])
roi_info_df = roi_info_df.reset_index(drop=True)
roi_info_df['ROI_ID'] = roi_info_df.index
roi_info_df['ROI_Name'] = ['ROI'+str(r).zfill(3) for r in roi_info_df['ROI_ID']]
print(roi_info_df.shape)
roi_info_df.to_csv(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv'), index=False)


# In[21]:


roi_info_df.reset_index(drop=True)


# # 9. We create a new NIFTI file where those ROIs have been removed.

# In[22]:


# Build 3dcalc expressions to isolate removed ROIs and generate an FOV-restricted atlas.
if len(bad_roi_list) > 0:
    bad_rois_minus = '-'.join([str(r)+'*equals(a,'+str(r)+')' for r,rs in bad_roi_list])
    bad_rois_plus  = '+'.join([str(r)+'*equals(a,'+str(r)+')' for r,rs in bad_roi_list])
    print(bad_rois_minus)
    print(bad_rois_plus)
else:
    print('No ROI needs to be removed')


# In[23]:


# Run AFNI commands to write removed-ROI and restricted-atlas NIfTI files.
if len(bad_roi_list) > 0:
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


# In[24]:


from nilearn.plotting import plot_roi


# In[25]:


plot_roi(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.nii.gz'),title='Original ATLAS')


# In[26]:


if len(bad_roi_list) > 0:
    plot_roi(osp.join(ATLAS_DIR,f'{ATLAS_NAME}.RemovedROIs.nii.gz'),title='ROIs that will be removed from the ATLAS')


# In[27]:


plot_roi(osp.join(ATLAS_DIR,f'rm.{ATLAS_NAME}.fov_restricted.nii.gz'),title='FOV-Restricted ATLAS')


# In[28]:


# Re-rank ROI labels so the restricted atlas has contiguous integer ROI IDs.
command = f"""ml afni; \
             cd {ATLAS_DIR}; \
             3dRank -overwrite -prefix {ATLAS_NAME}.nii.gz -input rm.{ATLAS_NAME}.fov_restricted.nii.gz;"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

