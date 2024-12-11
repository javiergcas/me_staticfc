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
# This notebook computes the average RSME across the whole brain for each scan and saves it as a text file.
#
# This would allow me to color scans by this amount, but we did not end up using it for the OHBM abstract.
#
# RSME could be a valuable diagnostic for scans that did not improve after the application of tedana.

import pandas as pd
import numpy as np
import os.path as osp
import subprocess
from tqdm import tqdm

from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, ATLAS_NAME, PRJ_DIR, CODE_DIR
ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
Nscans          = dataset_info_df.shape[0]
print('++ Number of scans: %s scans' % Nscans)
dataset_scan_list = list(dataset_info_df.index)
Nacqs = 201

for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    wdir       = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}')
    mask_path  = osp.join(wdir,f'mask_epi_anat.{sbj}+tlrc.HEAD')
    input_path = osp.join(wdir,'tedana_r01','rmse.nii.gz')
    out_path   = osp.join(wdir,'tedana_r01','rmse.avg.txt')
    command    = "ml afni; cd {wdir}; 3dROIstats -mask {mask_path} -quiet {input_path} > {out_path}".format(wdir=wdir, 
                                                                                                            mask_path=mask_path,
                                                                                                            input_path=input_path,
                                                                                                            out_path=out_path)
    output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
