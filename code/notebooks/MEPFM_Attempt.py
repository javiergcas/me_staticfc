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
import numpy as np
from tqdm import tqdm
import os.path as osp
import subprocess
import datetime
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, ATLAS_NAME, PRJ_DIR, CODE_DIR
ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
from nilearn.connectome import sym_matrix_to_vec
from sfim_lib.io.afni import load_netcc
import hvplot.pandas
import seaborn as sns
import holoviews as hv
import xarray as xr
import panel as pn
from itertools import combinations_with_replacement, combinations
from shutil import rmtree
import os

# # 1. Load Dataset Information

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
Nscans          = dataset_info_df.shape[0]
print('++ Number of scans: %s scans' % Nscans)
dataset_scan_list = list(dataset_info_df.index)
Nacqs = 201

echoes_dict = {'e01':13.7,'e02':30,'e03':47}

for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    wdir     = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}')
    e1_input = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.e01.volreg.scale.tproject_ALL+tlrc.HEAD')
    e2_input = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.e02.volreg.scale.tproject_ALL+tlrc.HEAD')
    e3_input = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.e03.volreg.scale.tproject_ALL+tlrc.HEAD')
    mask     = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','mask_tedana_at_least_one_echo.nii.gz')
    te1      = echoes_dict['e01']
    te2      = echoes_dict['e02']
    te3      = echoes_dict['e03']
    criteria = 'bic'
    hrf      = 'SPGM1'
    out_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'MEPFM_errts.{sbj}.r01.exx.volreg.scale.tproject_ALL')
    if osp.exists(out_path):
        rmtree(out_path)
    os.makedirs(out_path)
    command = """ml afni; \
                ml R; \
                cd {wdir}; \
                3dMEPFM -overwrite -input {e1_input} {te1} -input {e2_input} {te2} -input {e3_input} {te3} -criteria {criteria} -hrf SPMG1 -R2only -jobs 32 -prefix {out_path} -verb 1;""".format(wdir=wdir, 
                       e1_input=e1_input,e2_input=e2_input,e3_input=e3_input,
                       te1=te1,te2=te2,te3=te3, 
                       out_path=out_path,
                       criteria=criteria)
    dfgdg

command


