import sys
import argparse
from itertools import combinations, combinations_with_replacement
from nilearn.connectome import sym_matrix_to_vec
import numpy as np
import pandas as pd
import os.path as osp

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

def mse_dist(points,m1,s1,m2,s2,weight_fn=None, max_weight_fn=lambda r: np.minimum(r,np.quantile(r,.99)),tol=1e-12, verbose_return=False):
    """
    Compute the weighted fraction of points closer to line 1 vs line 2.
    Args:
        points: Nx2 array of (x,y) points (e.g., FC values from two echo pairs)
        m1: slope of line 1 
        s1: intercept of line 1
        m2: slope of line 2
        s2: intercept of line 2
        weight_fn: function to compute weights based on radius
        max_weight_fn: function to cap weights
        tol: tolerance for considering points as ties
        verbose_return: if True, return detailed results
    points: Nx2 array of (x,y) points
    """
    x  = points[:,0]; y = points[:,1]
    pd1 = compute_residuals(x,y,m1,s1)
    pd2 = compute_residuals(x,y,m2,s2)
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

    if verbose_return:
        return {'p_line1':frac_line1,'d1':pd1,'w':w,'r':r}
    else:
        return frac_line1

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
    
def process_command_line():
    parser = argparse.ArgumentParser(description="Compute pBOLD")
    parser.add_argument("-d", "--data", action="store", type=str, required=True, dest="data_paths", default=None, 
                        help="Path csv files containing the ROI timeseries per echo. Paths separated by commas")
    parser.add_argument("-e", "--echo_times", action="store", type=str, required=True, dest="echo_times", default=None,
                        help="Echo times in milliseconds, separated by commas")
    parser.add_argument("-m", "--fc_metric", action="store", type=str, required=False, dest="fc_metric", default='cov', choices=['cov','corr'],
                        help="Functional connectivity metric to use: covariance ('cov') or correlation ('corr')")
    parser.add_argument("-o", "--output", action="store", type=str, required=True, dest="output_path", default=None,
                        help="Output path for the pBOLD results")
    parser.add_argument("-t", "--line_pref_tolerance", required=False, dest="line_pref_tolerance", default=1e-3,
                        help="Tolerance for considering points as ties when computing line preference")
    parser.add_argument("-mrq", "--max_radius_quantile", required=False, dest="max_radius_quantile", default=0.95,
                        help="Maximum radius (in quantile terms) to apply when computing weighted preferences")
    return parser.parse_args()

def main():
    print("\n++ Parsing command line arguments...")
    print("++ ---------------------------------")
    opts = process_command_line()
    print('++ INFO: Line Preference Tolerance = %.6f' % opts.line_pref_tolerance)
    print('++ INFO: Maximum Radius Quantile   = %.2f' % opts.max_radius_quantile)
    
    # Extract number of echoes and of provided timeseries paths
    ts_paths     = opts.data_paths.split(',')
    num_ts_paths = len(ts_paths)
    tes          = [float(te) for te in opts.echo_times.split(',')]
    num_tes      = len(tes)
    tes_list     = ['e'+str(i).zfill(2) for i in range(1, num_tes+1)]
    tes_dict     = {te_label: te_value for te_label, te_value in zip(tes_list, tes)}

    assert num_ts_paths == num_tes, "++ INFO: Number of data paths must match number of echo times"
    print('++ INFO: Number of Echoes = %d' % num_tes)
    print('++ INFO: Echo Times (ms) = %s' % str(tes_dict))
    
    te_pairs_tuples = list(combinations_with_replacement(tes_list,2))
    te_pairs        = [('|').join(map(str,i)) for i in te_pairs_tuples]
    te_quadruples   = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(te_pairs_tuples,2)]
    
    # Load the timeseries data
    print('\n++ INFO: Loading timeseries data from provided paths...')
    print('++ ----------------------------------------------------')
    ts_data = {}
    for e, path in zip(tes_list, ts_paths):
        ts_data[e] = np.loadtxt(path)
        print('++ INFO: [%s] Loaded timeseries from %s with shape %s --> %s' % (e,path, str(ts_data[e].shape),str(ts_data[e][0,0:5])))
    
    # Compute FC matrices for all possible echo pairs
    print('\n++ INFO: Computing FC matrices for all possible echo pairs...')
    print('++ ----------------------------------------------------------')
    print(' + INFO: Available TE Pairs = %s' % str(te_pairs))

    fc_matrices = {}
    for e1,e2 in te_pairs_tuples:
        roi_ts_x      = ts_data[e1]
        roi_ts_y      = ts_data[e2]
        if opts.fc_metric == 'cov':
            aux = np.cov(roi_ts_x.T, roi_ts_y.T)[:roi_ts_x.shape[1], roi_ts_x.shape[1]:]
        if opts.fc_metric == 'corr':
            aux = np.corrcoef(roi_ts_x.T, roi_ts_y.T)[:roi_ts_x.shape[1], roi_ts_x.shape[1]:]
        print('++ INFO: Computed FC matrix for echo pair (%s, %s) has shape %s --> %s...' % (e1, e2, str(aux.shape), str(aux[0,0:5])))
        fc_matrices[(e1, e2)] = aux
    # Compute pBOLD for all possible echo quadruples
    print('\n++ INFO: Computing pBOLD for all possible echo quadruples...')
    print('++ ---------------------------------------------------------')
    print('++ INFO: TE Quadruples = %s' % str(te_quadruples))
    So_line_sl, So_line_int = 1.,0. # This is always the same
    BOLD_line_int = 0.              # This is always the same
    if opts.fc_metric  == 'corr':
        BOLD_line_sl = 1.
    pBOLD_results = {}
    for quad in te_quadruples:        
        print(' +       [%s]' % quad, end=' | ')
        eep1, eep2 = quad.split('_vs_')
        e1_X, e2_X = eep1.split('|')
        e1_Y, e2_Y = eep2.split('|')
        fc_X       = sym_matrix_to_vec(fc_matrices[(e1_X, e2_X)], discard_diagonal=True)
        fc_Y       = sym_matrix_to_vec(fc_matrices[(e1_Y, e2_Y)], discard_diagonal=True)
        fc_np      = np.vstack((fc_X, fc_Y)).T
        
        # Calculate intercept for cov (if needed)
        # =======================================
        if opts.fc_metric == 'cov':
            BOLD_line_sl  = (tes_dict[e1_Y]*tes_dict[e2_Y])/(tes_dict[e1_X]*tes_dict[e2_X])
        print('BOLD Slope = %.2f, Inter. = %.2f' % (BOLD_line_sl, BOLD_line_int), end=' -> ')
        pBOLD = mse_dist(fc_np, 
                        BOLD_line_sl, BOLD_line_int,
                        So_line_sl,   So_line_int, 
                        max_weight_fn=lambda r: np.minimum(r,np.quantile(r,opts.max_radius_quantile)),
                        tol=opts.line_pref_tolerance)
        print('pBOLD = %.4f' % (pBOLD))
        pBOLD_results[quad] = pBOLD
    pBOLD_df = pd.Series(pBOLD_results, name='pBOLD').to_frame()
    # Compute Single pBOLD value per scan
    # ===================================
    print('++ INFO: Computing scan-level pBOLD...')
    print('++ -----------------------------------')
    # a) Compute the chord weigths associated with each TE quadruple
    chord_weights = []
    for ppe in te_quadruples:
        eep1,eep2  = ppe.split('_vs_')
        e1_X, e2_X = eep1.split('|')
        e1_Y, e2_Y = eep2.split('|')
        BOLD_line_sl  = (tes_dict[e1_Y]*tes_dict[e2_Y])/(tes_dict[e1_X]*tes_dict[e2_X])
        chord_weights.append(chord_distance_between_intersecting_lines(1.0, BOLD_line_sl, r=0.5))
    chord_weights = np.array(chord_weights)
    print(' + Chord Weights = %s' % str(chord_weights))
    #) Compute scan-level pBOLD weigthed avarate fo all computed values
    pBOLD_scan_level = (pBOLD_df['pBOLD'].values * chord_weights).sum()/ chord_weights.sum()
    pBOLD_df.loc['scan'] = pBOLD_scan_level
    print(' +       Scan Level | pBOLD = %.4f' % (pBOLD_scan_level))
    # Save to disk
    # ============
    pBOLD_df.index.name='ee_vs_ee'
    pBOLD_df.to_csv(opts.output_path, index=True, header=True)
    print('++ INFO: Results saved to disk --> %s' % opts.output_path)

if __name__ == '__main__':
   sys.exit(main())