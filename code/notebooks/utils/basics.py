import os.path as osp
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement, combinations
import statsmodels.api as sm
import math

# Project folders of interest and file information
# ================================================
PRJ_DIR       = '/data/SFIMJGC_HCP7T/BCBL2024/'       # Project folder
ATLAS_NAME    = 'Schaefer2018_400Parcels_17Networks'  # Parcellation used in this project

PRCS_DATA_DIR   = osp.join(PRJ_DIR,'prcs_data')          # Folder where pre-processed data resides
ATLASES_DIR     = osp.join(PRJ_DIR,'atlases')            # Folder where atlases (original and modified) reside
CODE_DIR        = osp.join(PRJ_DIR,'me_staticfc','code') # Folder with scripts, programs and notebooks
DOWNLOAD_DIRS   = {'evaluation':osp.join(PRJ_DIR,'openeuro','des003592-download'), # Location of the Spreng dataset (as originally downloaded)
                   'discovery':osp.join(PRJ_DIR,'openeuro','meica_eval')}          # Location of the previously publised multi-echo, cardiac gated dataset

TES_MSEC = {'discovery':{'e01':13.9,'e02':31.7, 'e03':49.5},   # Echo Times for the discovery dataset in msec
            'evaluation':{'e01':13.7,'e02':30,'e03':47},}      # Echo Times for the evaluation dataset in msec
FMRI_TRS={'discovery':'2.5s',   # Repetion Time for the discovery dataset
     'evaluation':'3s'}         # Repetition Time for the evaluation dataset

FMRI_FINAL_NUM_SAMPLES={'discovery':192,   # Number of TRs after initial volume discarding in the discovery dataset
                        'evaluation':201}  # Number of TRs after initial volume discarding in the evaluation dataset

NUM_DISCARDED_VOLUMES = {'discovery':3.0,  # Number of discarded volumes in the discovery dataset
                         'evaluation':3.0} # Number of discarded volumes in the evaluation dataset

SESSIONS = {'discovery':['constant_gating','cardiac_gating'],  # Sessions in the discovery dataset
            'evaluation':['ses-1','ses-2']}                    # Sessions in the evaluation dataset


# List of unique echo time pairs and quadruples
# =============================================
echo_pairs_tuples   = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]

LABEL_MAPPING = {'ALL_Basic':'Basic','ALL_GS':'GSR','ALL_Tedana-fastica':'Tedana','ALL_NoRegression':'No Regression',
                 'KILL_Basic':'Basic w/ censoring','KILL_GS':'GSR w/ censoring','KILL_Tedana-fastica':'Tedana w/censoring','KILL_NoRegression':'No Regression w/censoring',
                 'NORDIC':'on'}

# Colormap for networks associated with the Power 264 atlas
# =========================================================
def get_altas_info(ATLAS_DIR, ATLAS_NAME):
    roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
    print("++ INFO [get_nw_cmap]: Gathering ROI information from file %s" % roi_info_path)
    roi_info_df   = pd.read_csv(roi_info_path)
    roi_info_df.head(5)

    Nrois = roi_info_df.shape[0]
    Ncons = int(((Nrois) * (Nrois-1))/2)

    print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))

    cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}
    return roi_info_df, cmap

# ====================================================================================================================================================

# Function to get list of scans available in each dataset
# -------------------------------------------------------
def get_dataset_index(dataset, verbose=True):
    """
    Input: dataset ID
    Output: multiindex with sbj and sess as levels
    """
    if dataset == 'discovery':
        sbj_list = ['MGSBJ01',  'MGSBJ02',  'MGSBJ03',  'MGSBJ04',  'MGSBJ05',  'MGSBJ06',  'MGSBJ07']
        ses_list = ['constant_gated', 'cardiac_gated']
        dataset_info_df = pd.DataFrame(index=pd.MultiIndex.from_product([sbj_list,ses_list],names=['Subject','Session']))
        out = dataset_info_df.index
    elif dataset == 'evaluation':
        dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
        dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
        out = dataset_info_df.index
    else:
        out = None
    if verbose:
        print('++ Number of scans    = %d' % out.shape[0])
        print('++ Number of subjects = %d' % out.get_level_values('Subject').get_level_values('Subject').unique().shape[0])
    return out

# ===============================
#      pBOLD-related functions
# ===============================
def compute_residuals(x, y, m, b):
    """Compute residuals after projection onto the line y = mx + b."""
    x_p, y_p = project_points(x, y, m, b)
    residuals = np.sqrt((x - x_p)**2 + (y - y_p)**2)
    return residuals

def project_points(x, y, m, b):
    """Project points (x, y) onto the line y = mx + b."""
    denom = 1 + m**2
    x_p = (x + m * (y - b)) / denom
    y_p = m * x_p + b
    return x_p, y_p

def mse_dist(points,m1,m2,weight_fn=None, max_weight_fn=lambda r: np.minimum(r,np.quantile(r,.99)),tol = 1e-12, verbose_return=False):
    x  = points[:,0]; y = points[:,1]
    pd1 = compute_residuals(x,y,m1,0.0)
    pd2 = compute_residuals(x,y,m2,0.0)
    r  = np.sqrt(x**2 + y**2)
    if weight_fn is None:
        weight_fn = lambda r: r   # linear weight by radius
    w = weight_fn(r)
    if max_weight_fn is not None:
        w = max_weight_fn(w)
    total_weight = w.sum()
    # Line 1
    pref1 = (pd1 < pd2).astype(float)
    ties = np.isclose(pd1, pd2, atol=tol)
    pref1[ties] = 0.5
    weighted_pref1 = (w * pref1).sum()
    frac_line1 = weighted_pref1 / (total_weight + 1e-16)

    # Line 2
    pref2 = (pd1 > pd2).astype(float)
    tol = 1e-12
    ties = np.isclose(pd1, pd2, atol=tol)
    pref2[ties] = 0.5
    weighted_pref2 = (w * pref2).sum()
    frac_line2 = weighted_pref2 / (total_weight + 1e-16)
    if verbose_return:
        return {'p_line1':frac_line1,
            'p_line2':frac_line2,
            'd1':pd1,
            'd2':pd2,
            'w':w,
            'r':r}
    else:
        return frac_line1,frac_line2

def chord_distance_between_intersecting_lines(m1, m2, r=1.0):
    """
    Compute the chord distance between two lines intersecting at the origin,
    based on points at distance r from the origin along each line.

    Inputs:
    m1: slope of the first line
    m2: slope of the second line
    r: radious at which to compute the distance [default = 1.0]

    Returns:
    distance: chord distance between both lines.
    """
    # points on line 1
    x1 = r / np.sqrt(1 + m1**2)
    y1 = m1 * x1

    # points on line 2
    x2 = r / np.sqrt(1 + m2**2)
    y2 = m2 * x2

    # Euclidean distance between the points
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

def line_circle_intersection(m, b, r=1.0):
    """
    Find the coordinates where the line y = m*x + b intersects
    the circle x^2 + y^2 = r^2 in the positive quadrant.

    Args:
        m (float): slope of the line
        b (float): intercept of the line
        r (float): radius of the circle (default = 1)

    Returns:
        (x, y) if an intersection exists in Q1
        None if no such intersection exists
    """
    # Quadratic coefficients: (1+m^2)x^2 + 2mb x + (b^2 - r^2) = 0
    A = 1 + m**2
    B = 2 * m * b
    C = b**2 - r**2

    # Discriminant
    D = B**2 - 4*A*C
    if D < 0:
        return None  # no real intersection

    sqrtD = math.sqrt(D)

    # Two possible solutions for x
    x1 = (-B + sqrtD) / (2*A)
    x2 = (-B - sqrtD) / (2*A)

    # Corresponding y
    y1 = m * x1 + b
    y2 = m * x2 + b

    candidates = [(x1, y1), (x2, y2)]

    # Return the one in the positive quadrant
    for (x, y) in candidates:
        if x >= 0 and y >= 0:
            return (x, y)

    return None  # no intersection in Q1

# Signal Processing Functions
# ===========================

def detrend_signal(y):
    """
    Removes mean and linear treand from a signal
    """
    nt    = len(y)
    trend = sm.add_constant(np.arange(nt))
    y_detrended = sm.OLS(y, trend).fit().resid
    return y_detrended

# AFNI IO Functions
# =================
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

# plotting related functions
# ==========================
def calculate_mean_deviation(data):
    """
    Calculates the mean absolute deviation (MAD) of a dataset.

    Args:
        data (list or numpy.ndarray): The input dataset.

    Returns:
        float: The mean absolute deviation of the data.
    """
    mean_value = np.mean(data)
    absolute_deviations = [abs(x - mean_value) for x in data]
    mean_deviation = np.mean(absolute_deviations)
    return mean_deviation