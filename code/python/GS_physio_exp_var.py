import sys
sys.path.append('../notebooks/')
from utils.basics import PRCS_DATA_DIR, PRJ_DIR
from utils.basics import detrend_signal

import argparse
import numpy as np
import pandas as pd
import os.path as osp

from scipy.stats import zscore
from afnipy.lib_afni1D import Afni1D
import statsmodels.api as sm
from tqdm import tqdm

def process_command_line():
    parser = argparse.ArgumentParser(description="Global Signal Kappa and Rho computation")
    parser.add_argument("-g", "--gs_path",       action="store", type=str, required=True, dest="gs_path",      default=None, help="Path to global signal")
    parser.add_argument("-p", "--slibase_path",  action="store", type=str, required=True, dest="slibase_path", default=None, help="Path to physiological regressors")
    parser.add_argument("-o", "--output_path",   action="store", type=str, required=True, dest="output_path",  default=None, help="Path to output file")
    return parser.parse_args()

def main():
    opts = process_command_line()
    print('++ INFO: Global Signal Path                           = %s' % opts.gs_path)
    print('++ INFO: Physiological Regressors (slibase file) Path = %s' % opts.slibase_path)
    print('++ INFO: Output Path                                  = %s' % opts.output_path)
    gs_path      = opts.gs_path
    slibase_path = opts.slibase_path
    output_path  = opts.output_path

    # Check Inputs
    # ============
    if not osp.exists(gs_path):
       print("++ ERROR: I cannot find the global signal [%s]" % gs_path)
       print("          Program will exit now.")
       return -1

    if not osp.exists(slibase_path):
       print("++ ERROR: I cannot find the physio regressors file [%s]" % slibase_path)
       print("          Program will exit now.")
       return -1

    output_dir = osp.dirname(output_path)
    if not osp.exists(output_dir):
       print("++ ERROR: Output folder does not exists [%5]" % output_dir)
       print("          Create it and try again.")
       return -1

    # Load the Global Signal
    # ======================
    print("++ INFO: Loading GS Timeseries (and generate detrended version)...")
    gs_df         = pd.DataFrame(np.loadtxt(gs_path))
    gs_df.columns = ['orig']
    print("++       Detrend the GS timeseries")
    gs_df['det']  = detrend_signal(gs_df['orig'].values.squeeze())
    print('+        GS dataframe shape is %s' % str(gs_df.shape))

    # Load Physiological Regressors
    # =============================
    print("++ INFO: Loading Physiological regressors into memory...",end="")
    slibase_obj  = Afni1D(slibase_path)
    slibase_df   = pd.read_csv(slibase_path, comment='#', delimiter=' +', header=None, engine='python')
    slibase_df.columns = slibase_obj.labels
    slibase_df         = slibase_df[3::].reset_index(drop=True)
    slibase_det_df     = slibase_df.copy()
    print('[DONE]')
    print('++ INFO: Detrending (mean and linear trend) the physio regressors')
    for c in tqdm(slibase_det_df.columns):
        slibase_det_df[c] = detrend_signal(slibase_df[c].values.squeeze())

    # Select Optimal shifted version of physio regressors
    # ===================================================
    print("++ INFO: Finding time-shifted version with optimal correlation towards GS")
    phys_reg_list = np.unique([c.split('.',1)[1] for c in slibase_det_df.columns])
    corrs_with_physio = pd.concat([gs_df,slibase_det_df],axis=1).corr().loc['det']
    sel_physio_regs = []
    for pr in phys_reg_list:
       aux = (corrs_with_physio.loc[[i for i in corrs_with_physio.index if pr in i]].abs().sort_values(ascending=False)).index[0]
       sel_physio_regs.append(aux)
    print(" +      Selected Regressors: %s" % str(sel_physio_regs))

    # Compute R2 and R2adj
    # ====================
    print("++ INFO: Estimating the variance explained...")
    y = gs_df['det']
    X = slibase_det_df[sel_physio_regs]
    X = sm.add_constant(X)
    model = sm.OLS(y,X).fit()
    print(model.summary())
    print('====================================================')
    print('++ INFO: Variance Explained = %.2f' % model.rsquared)

    # Save Model to disk
    # ==================
    print("++ INFO: Saving model to disk [%s]" % output_path)
    model.save(output_path)
    
if __name__ == '__main__':
   sys.exit(main())
