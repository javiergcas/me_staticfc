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

TES_MSEC = {'discovery':{'e01':13.9,'e02':31.7, 'e03':49.5},
            'evaluation':{'e01':13.7,'e02':30,'e03':47},}

NUM_DISCARDED_VOLUMES = {'evaluation':3.0}

SESSIONS = {'discovery':['constant_gating','cardiac_gating'],
            'evaluation':['ses-1','ses-2']}

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

def get_dataset_index(dataset, verbose=True):
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

# def mse_dist(points,m1,m2,weight_fn=None, max_weight=None,tol = 1e-12):
#     x  = points[:,0]; y = points[:,1]
#     pd1 = compute_residuals(x,y,m1,0.0)
#     pd2 = compute_residuals(x,y,m2,0.0)
#     r  = np.sqrt(x**2 + y**2)
#     if weight_fn is None:
#         weight_fn = lambda r: r   # linear weight by radius
#     w = weight_fn(r)
#     if max_weight is not None:
#         w = np.minimum(max_weight,w)
#     total_weight = w.sum()

#     # Line 1
#     pref1 = (pd1 < pd2).astype(float)
#     ties = np.isclose(pd1, pd2, atol=tol)
#     pref1[ties] = 0.5
#     weighted_pref1 = (w * pref1).sum()
#     frac_line1 = weighted_pref1 / (total_weight + 1e-16)

#     # Line 2
#     pref2 = (pd1 > pd2).astype(float)
#     tol = 1e-12
#     ties = np.isclose(pd1, pd2, atol=tol)
#     pref2[ties] = 0.5
#     weighted_pref2 = (w * pref2).sum()
#     frac_line2 = weighted_pref2 / (total_weight + 1e-16)
#     return frac_line1,frac_line2
    
# Alternative metrics
def angdiff(a, b):
    """Smallest signed angle difference a-b in [-pi, pi]."""
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

def ang_dist(points,m1,m2,weight_fn=None, max_weight=None,tol = 1e-12):
    """
    points: (N,2) array of (x,y)
    m1, m2: slopes of the two lines through origin
    weight_fn: function r -> weight (if None uses w = r)
    returns: dict with weighted counts / fraction preferring line1
    """
    x = points[:,0]; y = points[:,1]
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    if weight_fn is None:
        weight_fn = lambda r: r   # linear weight by radius
    w = weight_fn(r)
    if max_weight is not None:
        w = np.minimum(max_weight,w)
    total_weight = w.sum()
    
    phi1 = np.arctan(m1)
    phi2 = np.arctan(m2)
    d1 = np.abs(angdiff(theta, phi1))
    d2 = np.abs(angdiff(theta, phi2))
    
    # Line 1
    pref1 = (d1 < d2).astype(float)
    ties = np.isclose(d1, d2, atol=tol)
    pref1[ties] = 0.5
    weighted_pref1 = (w * pref1).sum()
    frac_line1 = weighted_pref1 / (total_weight + 1e-16)

    # Line 1
    pref2 = (d1 > d2).astype(float)
    ties = np.isclose(d1, d2, atol=tol)
    pref2[ties] = 0.5
    weighted_pref2 = (w * pref2).sum()
    frac_line2 = weighted_pref2 / (total_weight + 1e-16)
    return frac_line1, frac_line2

    
    
# def weighted_line_preference(points, m1, m2, weight_fn=None):
#     """
#     points: (N,2) array of (x,y)
#     m1, m2: slopes of the two lines through origin
#     weight_fn: function r -> weight (if None uses w = r)
#     returns: dict with weighted counts / fraction preferring line1
#     """
#     x = points[:,0]; y = points[:,1]
#     theta = np.arctan2(y, x)
#     r = np.sqrt(x**2 + y**2)
#     if weight_fn is None:
#         weight_fn = lambda r: r   # linear weight by radius
#     w = weight_fn(r)
#     phi1 = np.arctan(m1)
#     phi2 = np.arctan(m2)
#     d1 = np.abs(angdiff(theta, phi1))
#     d2 = np.abs(angdiff(theta, phi2))
#     # prefer the line with smaller angular error
#     pref1 = (d1 < d2).astype(float)
#     # tie-breaker: if equal within tolerance, split 0.5
#     tol = 1e-12
#     ties = np.isclose(d1, d2, atol=tol)
#     pref1[ties] = 0.5
#     weighted_pref1 = (w * pref1).sum()
#     total_weight = w.sum()
#     frac_line1 = weighted_pref1 / (total_weight + 1e-16)
#     return {
#         'd1':d1,
#         'd2':d2,
#         'frac_line1': frac_line1,
#         'weighted_pref1': weighted_pref1,
#         'total_weight': total_weight,
#         'per_point_weights': w,
#         'per_point_pref1': pref1
#     }

def em_angle_mixture(points, m1, m2, max_iter=100, tol=1e-6, weight_fn=None):
    """
    EM for a 2-component mixture on angles with weights based on radius.
    Returns responsibilities (probability of belonging to component 1),
    and estimated mixing proportion pi, sigmas.
    """
    x = points[:,0]; y = points[:,1]
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    if weight_fn is None:
        weight_fn = lambda r: r
    w = weight_fn(r)
    phi1 = np.arctan(m1)
    phi2 = np.arctan(m2)
    N = len(theta)
    # init responsibilities by hard weighted preference
    d1 = np.abs(angdiff(theta, phi1)); d2 = np.abs(angdiff(theta, phi2))
    resp1 = (d1 < d2).astype(float)
    # small random jitter to avoid degeneracy
    resp1 = resp1 * 0.9 + 0.1 * np.random.rand(N)
    pi = resp1.mean()
    # initialize sigmas:
    def weighted_var(diff, rts):
        return np.sqrt((rts * (diff**2)).sum() / (rts.sum() + 1e-16))
    sigma1 = weighted_var(angdiff(theta, phi1), w * resp1)
    sigma2 = weighted_var(angdiff(theta, phi2), w * (1-resp1))
    sigma1 = max(sigma1, 1e-3); sigma2 = max(sigma2, 1e-3)

    for it in range(max_iter):
        # E-step: compute weighted Gaussian likelihoods
        # p ~ exp(-0.5 * w * (diff/sigma)^2)  (w acts like precision scaling)
        a1 = -0.5 * w * (angdiff(theta, phi1)**2) / (sigma1**2)
        a2 = -0.5 * w * (angdiff(theta, phi2)**2) / (sigma2**2)
        # to avoid underflow, shift
        A = np.vstack([a1, a2])
        A = A - A.max(axis=0)
        lik1 = np.exp(A[0])
        lik2 = np.exp(A[1])
        # responsibilities
        numer1 = pi * lik1
        numer2 = (1-pi) * lik2
        denom = numer1 + numer2 + 1e-16
        new_resp1 = numer1 / denom
        # M-step
        new_pi = new_resp1.mean()
        # update sigmas using weighted residuals
        s1 = weighted_var(angdiff(theta, phi1), w * new_resp1)
        s2 = weighted_var(angdiff(theta, phi2), w * (1-new_resp1))
        # stabilize
        s1 = max(s1, 1e-6); s2 = max(s2, 1e-6)
        # check convergence
        if np.abs(new_pi - pi) < tol and np.abs(s1 - sigma1) < tol and np.abs(s2 - sigma2) < tol:
            pi, sigma1, sigma2, resp1 = new_pi, s1, s2, new_resp1
            break
        pi, sigma1, sigma2, resp1 = new_pi, s1, s2, new_resp1

    return {
        'pi': pi,                 # mixing proportion for line1
        'sigma1': sigma1,
        'sigma2': sigma2,
        'resp1': resp1,           # per-point P(component=1)
        'weights': w
    }


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
