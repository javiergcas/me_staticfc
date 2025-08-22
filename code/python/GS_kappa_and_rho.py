
import sys
sys.path.append('../notebooks/')
from utils.basics import PRCS_DATA_DIR, TES_MSEC, PRJ_DIR

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import os.path as osp

from scipy.stats import zscore
from tedana.metrics.collect import generate_metrics
from tedana.io import OutputGenerator
from tqdm import tqdm

def process_command_line():
    parser = argparse.ArgumentParser(description="Global Signal Kappa and Rho computation")
    parser.add_argument("-s", "--subject",  action="store", type=str, required=True, dest="sbj", default=None, help="Subject ID")
    parser.add_argument("-r", "--session",  action="store", type=str, required=True, dest="ses", default=None, help="Session ID")
    return parser.parse_args()

def main():
    opts = process_command_line()
    print('++ INFO: Subject = %s' % opts.sbj)
    print('++ INFO: Session = %s' % opts.ses)
    out_path = osp.join(PRCS_DATA_DIR,opts.sbj,f'D03_Preproc_{opts.ses}_NORDIC-off',f'{opts.sbj}_{opts.ses}_GS_kappa_and_rho.txt')
    print('++ INFO: Output Path = %s' % out_path)
    # Echo Times
    tes = list(TES_MSEC['evaluation'].values())
    ne  = len(tes)
    print('++ INFO: Echo Times in ms: %s' % str(tes))

    # Load the adaptive mask
    print('++ INFO: Loading adaptive mask...',end='')
    mask_path = osp.join(PRCS_DATA_DIR,opts.sbj,f'D03_Preproc_{opts.ses}_NORDIC-off','tedana_fastica','adaptive_mask.nii.gz')
    mask_img  = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    nx,ny,nz  = mask_data.shape
    mask_vec  = mask_data.reshape(nx*ny*nz,).astype(int)
    print('[DONE]')
    print('         Mask Shape %s --> %s' % (str(mask_data.shape),str(mask_vec.shape)))

    # Extract number of acquisitions from first echo
    print('++ INFO: Extracting number of acquistions...', end='')
    e1_path = osp.join(PRCS_DATA_DIR,opts.sbj,f'D03_Preproc_{opts.ses}_NORDIC-off',f'pb03.{opts.sbj}.r01.e01.volreg+tlrc.HEAD')
    e1_img  = nib.load(e1_path)
    e1_data = e1_img.get_fdata()
    _,_,_,nt = e1_data.shape
    print('%s time points.' % nt)

    # Load individual echo data
    print("++ INFO: Loading individual echo datasets...")
    data_cat = np.zeros((nx*ny*nz,ne,nt))
    for e,ee in enumerate(tqdm(list(TES_MSEC['evaluation'].keys()))):
       path = osp.join(PRCS_DATA_DIR,opts.sbj,f'D03_Preproc_{opts.ses}_NORDIC-off',f'pb03.{opts.sbj}.r01.{ee}.volreg+tlrc.HEAD')
       img  = nib.load(path)
       data = img.get_fdata()
       data_cat[:,e,:] = data.reshape(nx*ny*nz,nt)
    print(" +       data_cat.shape = %s" % str(data_cat.shape))

    # Load the Global Signal
    print("++ INFO: Loading GS Timeseries...")
    gs_path = osp.join(PRCS_DATA_DIR,opts.sbj,f'D03_Preproc_{opts.ses}_NORDIC-off', f'pb06.{opts.sbj}.r01.tedana_fastica_OC.GS.demean.1D') #f'pb03.{opts.sbj}.r01.e02.volreg.GS.demean.1D')
    print(' +       GS Path = %s' % gs_path)
    if osp.exists(gs_path):
       gs      = zscore(np.loadtxt(gs_path)).reshape(nt,1)
    else:
       print('        Error: GS file [%s] not available' % gs_path)
       return -1
    print('+        GS shape is %s' % str(gs.shape))

    # Load OC Dataset
    print("++ INFO: Loading optimally combined dataset...")
    oc_path = osp.join(PRCS_DATA_DIR,opts.sbj,f'D03_Preproc_{opts.ses}_NORDIC-off','tedana_fastica','ts_OC.nii.gz')
    oc_img  = nib.load(oc_path)
    oc_data = oc_img.get_fdata()
    data_optcom = oc_data.reshape(nx*ny*nz,nt)
    print(" +       data_optcom.shape=%s" % str(data_optcom.shape))

    # Compute Metrics
    # Create Output Generator Object that will write nothing
    print("++ INFO: Computing Kappa and Rho for the Global Signal...",end='')
    io_generator = OutputGenerator(reference_img=mask_img,
        convention='orig',
        out_dir=osp.join(PRJ_DIR,'temp'),
        prefix=f'temp.{opts.sbj}.{opts.ses}',
        config="auto",
        overwrite=True,
        make_figures=False,
        verbose=False)
    # Compute Kappa and Rho
    component_table, mixing = generate_metrics(data_cat=data_cat,
        data_optcom=data_optcom, 
        mixing=gs,
        adaptive_mask=mask_vec,
        tes=tes,
        io_generator=io_generator,
        label='GS',
        metrics=['kappa','rho'])
    print("[DONE]")
    print(component_table)

    # Saving Results to disk
    print("++ INFO: Saving results to disk [%s]" % out_path)
    component_table.to_csv(out_path)

if __name__ == '__main__':
   sys.exit(main())
