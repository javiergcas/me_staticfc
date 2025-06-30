import os.path as osp
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement, combinations
import statsmodels.api as sm

PRJ_DIR       = '/data/SFIMJGC_HCP7T/BCBL2024/'
ATLAS_NAME    = 'Schaefer2018_400Parcels_17Networks'

PRCS_DATA_DIR = osp.join(PRJ_DIR,'prcs_data')
ATLASES_DIR   = osp.join(PRJ_DIR,'atlases')
CODE_DIR      = osp.join(PRJ_DIR,'me_staticfc','code')
SPRENG_DOWNLOAD_DIR = osp.join(PRJ_DIR,'openeuro','des003592-download')

TES_MSEC = {'Gating':{'e01':13.9,'e02':31.7, 'e03':49.5},
            'Spreng_Scanner1':{'e01':13.7,'e02':30,'e03':47},
            }

NUM_DISCARDED_VOLUMES = {'Spreng_Scanner1':3.0}

SESSIONS = {'Gating':['constant_gating','cardiac_gating'],
            'Spreng_Scanner1':['ses-1','ses-2']}

echo_pairs_tuples   = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]

# Echo Time information for the Spreng Dataset
#TES_MSEC_PER_SCANNER = {'1':{'e01':13.7,'e02':30,'e03':47},
#                        '2':{'e01':14,'e02':29.96,'e03':45.92}}
#
# We are only working with data from Site 1
#TES_MSEC = TES_MSEC_PER_SCANNER['1']

Power264_cmap = {''}

def detrend_signal(y):
    """
    Removes mean and linear treand from a signal
    """
    nt    = len(y)
    trend = sm.add_constant(np.arange(nt))
    y_detrended = sm.OLS(y, trend).fit().resid
    return y_detrended

def read_group_physio_reports(path):
    """
    Reads the output of command gen_ss_review_table when
    used on the outputs of phys_calc
    """
    df = pd.read_csv(path, sep='\t', header=[0,1] )
    new_columns = []
    current_lvl1 = None
    # Deal with empty labels in second level of column name
    for lvl1, lvl2 in df.columns:
        if "Unnamed" in str(lvl1):
            new_columns.append((current_lvl1, lvl2))
        else:
            current_lvl1 = lvl1
            new_columns.append((lvl1, lvl2))
    
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    # Add index based on Subject and Run
    df.index = pd.MultiIndex.from_tuples([v.split('_') for v in df[('subject','value')]], names=['Subject','Run'])
    return df
    
def read_gen_ss_review_table(file_path):
    """
    Reads the output of AFNI command gen_ss_review_table and
    organizes the contents into a pandas dataframe with meaningful
    column names
    
    Input
    -----
    file_path: path to input file
    
    Returns
    -------
    df: dataframe with the information in the file
    """
    if not osp.exists(file_path):
        print('++ERROR [read_gen_ss_review_table]: input file does not exists')
        return None
    df = pd.read_csv(file_path, sep='\t', header=None)
    original_columns  = [(a,b) for a,b in df.loc[[0,1]].values.T]
    df.drop([0,1], inplace=True)
    new_columns = []
    for (a,b) in original_columns:
        if a=='infile':
            new_columns = new_columns + ['infile']
            continue
        if (a=='echo times'):
            new_columns = new_columns + ['e01','e02','e03']
            continue
        if (a is np.nan):
            continue
        if (a=='orig voxel counts'):
            new_columns = new_columns + ['Nx','Ny','Nz']
            continue
        if (a=='orig voxel resolution'):
            new_columns = new_columns + ['orig Dx','orig Dy','orig Dz']
            continue
        if (a=='final voxel resolution'):
            new_columns = new_columns + ['final Dx','final Dy','final Dz']
            continue
        if (a=='orig volume center'):
            new_columns = new_columns + ['orig volume center x','orig volume center y','orig volume center z']
            continue
        new_columns = new_columns + [a]
    df.columns = new_columns
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            df[c] = df[c]
    print("++ INFO [read_gen_ss_review_table]: Number of scans = %d | Number of metrics per scan = %d" % (df.shape))
    return df

# QA-related functions

def project_points(x, y, m, b):
    """Project points (x, y) onto the line y = mx + b."""
    denom = 1 + m**2
    x_p = (x + m * (y - b)) / denom
    y_p = m * x_p + b
    return x_p, y_p

def compute_residuals(x, y, m, b):
    """Compute residuals after projection onto the line y = mx + b."""
    x_p, y_p = project_points(x, y, m, b)
    residuals = np.sqrt((x - x_p)**2 + (y - y_p)**2)
    return residuals

def softmax(x, substract_max=False):
  """
  Computes the softmax function for a given input vector x.

  Args:
    x: A NumPy array representing the input vector.

  Returns:
    A NumPy array representing the softmax output.
  """
  if substract_max:
      exp_x = np.exp(x - np.max(x)) # Subtracting max(x) for numerical stability
  else:
      exp_x = np.exp(x)
  return exp_x / np.sum(exp_x)
