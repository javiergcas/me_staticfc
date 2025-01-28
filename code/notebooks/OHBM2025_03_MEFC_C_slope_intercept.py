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
# This notebook will compute C-FC across all possible echo pairs both following Basic and Advanced denoising.
#
# It will then compute linear fits for the contrast between those.
#
# Finally, it generates summary figures of how the slope and intercept adheres to the situations when BOLD or non-BOLD alone dominate the data

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
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# +
from bokeh.models import Arrow, NormalHead, OpenHead, Range1d, Line, ColumnDataSource, HoverTool, ColorBar
#from bokeh.palettes import Muted3 as color
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper
#from bokeh.colors import RGB
from bokeh.palettes import Inferno256

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb2hex


# -

def value_to_color(value, vmin=-0.004, vmax=0.004, palette=Inferno256):
    """
    Map a value to a color using the seismic colormap.
    
    Parameters:
    - value (float): The value to map to a color.
    - vmin (float): Minimum value in the range.
    - vmax (float): Maximum value in the range.
    - palette (list): A Bokeh palette to use for coloring.
    
    Returns:
    - color (str): Hexadecimal representation of the color.
    """
    # Create the LinearColorMapper
    color_mapper = LinearColorMapper(palette=palette, low=vmin, high=vmax)
    
    # Map the value to the palette index
    normalized_value = (value - vmin) / (vmax - vmin) * (len(palette) - 1)
    clamped_index = max(0, min(len(palette) - 1, int(round(normalized_value))))
    
    # Get the corresponding color from the palette
    return palette[clamped_index]


# +
# Step 2: Convert the matplotlib colormap to a Bokeh palette
def mpl_to_bokeh_palette(mpl_cmap, n_colors=256):
    """Convert a matplotlib colormap to a Bokeh palette."""
    return [mpl_cmap(i / (n_colors - 1)) for i in range(n_colors)]

# Function to get color for a given value
def matplotlib_color_mapper(value, vmin=-0.04,vmax=0.04):
    # Create a colormap and normalizer
    seismic_cmap = plt.cm.seismic_r
    norm = Normalize(vmin=vmin, vmax=vmax)
    if value < vmin:
        value = vmin
    if value > vmax:
        value = vmax
    #if value < -0.04 or value > 0.04:
    #    raise ValueError("Value out of range [-0.04, 0.04]")
    rgba_color = seismic_cmap(norm(value))
    # Convert RGBA to Hex if needed
    return matplotlib.colors.to_hex(rgba_color)



# -

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# # 1. Load Dataset Information

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
Nscans          = dataset_info_df.shape[0]
print('++ Number of scans: %s scans' % Nscans)
dataset_scan_list = list(dataset_info_df.index)
Nacqs = 201

# # 2. Load Atlas Information

# +
roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df.head(5)

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)

print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))
# -

# Compute Euclidean Distance between ROI centroids

# +
# Select the columns that correspond to position
roi_coords_df = roi_info_df.set_index(['ROI_Name'])[['pos_R','pos_A','pos_S']]

# Convert the DataFrame to a NumPy array
roi_coords = roi_coords_df.values

# Calculate the Euclidean distance using broadcasting
roi_distance_matrix = np.sqrt(((roi_coords[:, np.newaxis] - roi_coords) ** 2).sum(axis=2))

# Convert to DataFrame
roi_distance_df = pd.DataFrame(roi_distance_matrix, index=roi_coords_df.index, columns=roi_coords_df.index)
# -

roi_distance_vect = sym_matrix_to_vec(roi_distance_df.values, discard_diagonal=True)

# # 3. Create lists of echo pairs, combinations of these, and all scan names
#
# Create list of all echo combinations and combinations of those

echo_pairs_tuples   = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]
print('Echo Pairs[n=%d]=%s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d]=%s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

# Create list of scan names as ```sbj-xx_ses-y```

scan_names = ['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list]

# Create list of all pairs of pairs of TEs (e.g., ```e01|e01_vs_e01|e03```)and also calculate their ideal BOLD slope based on the TEs.

echoes_dict = {'e01':13.7,'e02':30,'e03':47}
#echoes_dict = {'e01':14,'e02':29.96,'e03':45.92}
ideal_slopes = {}
for p in pairs_of_echo_pairs:
    x,y = p.split('_vs_')
    x_e1,x_e2 = x.split('|')
    y_e1,y_e2 = y.split('|')
    ideal_slopes[p] = (echoes_dict[y_e1] * echoes_dict[y_e2]) / (echoes_dict[x_e1] * echoes_dict[x_e2])
print(ideal_slopes)

# # 4. Load Basic Quality Information for each scan
#
# ## 4.1. Fraction of censored datapoints

mot_df = pd.DataFrame(index=scan_names,columns=['Percent Censored'])
mot_df.index.name = 'scan'
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    scan_name = '_'.join((sbj,ses))
    censor_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'motion_{sbj}_censor.1D')
    censor     = np.loadtxt(censor_path).astype(bool)
    mot_df.loc[scan_name,'Percent Censored'] = 100*(len(censor)-np.sum(censor))/len(censor)
    mot_df.loc[scan_name,'Percent Used']     = 100*(np.sum(censor))/len(censor)

# ## 4.2. Fraction of BOLD-like vs. non-BOLD like data

tedana_df = pd.DataFrame(index=scan_names,columns=['Var. likely-BOLD','Var. unlikely-BOLD','Var. accepted','Var. rejected'])
tedana_df.index.name = 'scan'
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    scan_name = '_'.join((sbj,ses))
    ica_table_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','tedana_r01','ica_metrics.tsv')
    ica_table = pd.read_csv(ica_table_path,sep='\t')
    tedana_df.loc[scan_name,'Var. likely-BOLD'] = ica_table.set_index(['classification_tags']).loc['Likely BOLD','variance explained'].sum()
    tedana_df.loc[scan_name,'Var. unlikely-BOLD'] = ica_table.set_index(['classification_tags']).loc['Unlikely BOLD','variance explained'].sum()
    tedana_df.loc[scan_name,'Var. accepted'] = ica_table.set_index(['classification']).loc['accepted','variance explained'].sum()
    tedana_df.loc[scan_name,'Var. rejected'] = ica_table.set_index(['classification']).loc['rejected','variance explained'].sum()

# ## 4.3. RMSE

rsme_df = pd.DataFrame(index=scan_names,columns=['Avg. RSME'])
rsme_df.index.name = 'scan'
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    scan_name = '_'.join((sbj,ses))
    rsme_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','tedana_r01','rmse.avg.txt')
    rsme = np.loadtxt(rsme_path)
    rsme_df.loc[scan_name,'Avg. RSME'] = rsme

# # 5. Pearson's FC Slope and Intercept
# ## 5.1. Load the data following Deoniosing Mode 1 (e.g, Basic, MEICA)
#
# We will create an xr.DataArray that will hold the slope and intercept of contrasting all 15 FC matrices for each scan separately. 
#
# We wll also then compute the averages per scan, so that we can characterize a given scan in 2D space.

# > **NOTE**: Here is where we select what particular comparison we want to check

# Basic Denoising vs. MEICA Denosing
x_data,x_scenario,y_data,y_scenario, x_label, y_label = 'volreg',  'ALL','meica_dn','ALL','Basic','MEICA'
# Basic Denoising vs. Basic Denoising + Global Signal Regression
#x_data,x_scenario,y_data,y_scenario, x_label, y_label = 'volreg','ALL','volreg','ALL_GSasis','Basic','Basic + GSR (asis)'
slope_inter_xr_all        = {}

# Load the ROI timeseries for the corresponding echoes, then compute the covariance matrices.
#
# Once all of them are available, let's compute the slope and intercept for all pairs of pairs

# %%time
slope_inter_xr_all[x_label] = xr.DataArray(dims=['scan','echo_pairing','statistic'],
                        coords={'scan':['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list],
                                'echo_pairing':pairs_of_echo_pairs,
                                'statistic':['Slope','Intercept']})
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    fc_xr_all       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    # Compute all covariance matrices for this scan
    for (e_x,e_y) in echo_pairs_tuples:
        roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.{x_data}.scale.tproject_{x_scenario}.{ATLAS_NAME}_000.netts')
        roi_ts_x      = np.loadtxt(roi_ts_path_x)
        roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.{x_data}.scale.tproject_{x_scenario}.{ATLAS_NAME}_000.netts')
        roi_ts_y      = np.loadtxt(roi_ts_path_y)
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = aux_r_v
    # Contract pairs of echoes against each other
    for pair_of_pairs in pairs_of_echo_pairs:
        p1,p2=pair_of_pairs.split('_vs_')
        x = fc_xr_all.sel(pair=p1)
        y = fc_xr_all.sel(pair=p2)
        slope, intercept = np.polyfit(x,y,1)
        slope_inter_xr_all[x_label].loc['_'.join((sbj,ses)),pair_of_pairs,'Slope'] = slope
        slope_inter_xr_all[x_label].loc['_'.join((sbj,ses)),pair_of_pairs,'Intercept'] = intercept

# ## 5.2. Load data and compute slope and intercept following Advanced denoising
#
# Same as the cell above, but this time for the second denoising pipeline

# %%time
slope_inter_xr_all[y_label] = xr.DataArray(dims=['scan','echo_pairing','statistic'],
                        coords={'scan':['_'.join((sbj,ses)) for sbj,ses in dataset_scan_list],
                                'echo_pairing':pairs_of_echo_pairs,
                                'statistic':['Slope','Intercept']})
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    fc_xr_all       = xr.DataArray(dims=['pair','edge'],
                      coords={'pair':  echo_pairs,
                              'edge':  np.arange(Ncons)})
    # Compute all covariance matrices for this scan
    for (e_x,e_y) in echo_pairs_tuples:
        roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.{y_data}.scale.tproject_{y_scenario}.{ATLAS_NAME}_000.netts')
        roi_ts_x      = np.loadtxt(roi_ts_path_x)
        roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.{y_data}.scale.tproject_{y_scenario}.{ATLAS_NAME}_000.netts')
        roi_ts_y      = np.loadtxt(roi_ts_path_y)
        aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
        aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
        # Compute the full correlation matrix between aux_ts_x and aux_ts_y
        aux_r   = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
        aux_r_v = sym_matrix_to_vec(aux_r, discard_diagonal=True)
        
        fc_xr_all.loc['|'.join((e_x,e_y)),:] = aux_r_v
    # Contract pairs of echoes against each other
    for pair_of_pairs in pairs_of_echo_pairs:
        p1,p2=pair_of_pairs.split('_vs_')
        x = fc_xr_all.sel(pair=p1)
        y = fc_xr_all.sel(pair=p2)
        slope, intercept = np.polyfit(x,y,1)
        slope_inter_xr_all[y_label].loc['_'.join((sbj,ses)),pair_of_pairs,'Slope'] = slope
        slope_inter_xr_all[y_label].loc['_'.join((sbj,ses)),pair_of_pairs,'Intercept'] = intercept

# # 6. Compute DBOLD
#
# This is the quality metric we use to see how much we are approaching to the ideal scenario of data being dominated only by BOLD fluctuations.
#
# For fun, we also compute the equivalent towards the non-BOLD ideal point (D_nonBOLD)

# %%time
df = pd.DataFrame(index=scan_names,columns=['dist_BOLD_'+x_label,'dist_BOLD_'+y_label,'dist_NonBOLD_'+x_label,'dist_NonBOLD_'+y_label])
for scan_name in tqdm(scan_names):
    dist_BOLD_X, dist_NonBOLD_X = [],[]  # List to hold DBOLD and DnonBOLD for denoising scenario 1 (x)
    dist_BOLD_Y, dist_NonBOLD_Y = [],[]  # List to hold BOLD and DnonBOLD for denoising scenario 2 (y)
    for pair_of_pairs in pairs_of_echo_pairs:
        BOLD_ideal_slope    = ideal_slopes[pair_of_pairs]   # Extract what is the ideal BOLD Slope for this TEs comparison
        NonBOLD_ideal_slope = 1                             # The ideal nonBOLD Slope is always 1.
        dist_BOLD_X = dist_BOLD_X + [euclidean([slope_inter_xr_all[x_label].loc[scan_name,pair_of_pairs,'Intercept'].values,
                                               slope_inter_xr_all[x_label].loc[scan_name,pair_of_pairs,'Slope'].values],
                                              [0,BOLD_ideal_slope])]
        dist_BOLD_Y = dist_BOLD_Y + [euclidean([slope_inter_xr_all[y_label].loc[scan_name,pair_of_pairs,'Intercept'].values,
                                               slope_inter_xr_all[y_label].loc[scan_name,pair_of_pairs,'Slope'].values],
                                              [0,BOLD_ideal_slope])]        
        dist_NonBOLD_X = dist_NonBOLD_X + [euclidean([slope_inter_xr_all[x_label].loc[scan_name,pair_of_pairs,'Intercept'].values,
                                                     slope_inter_xr_all[x_label].loc[scan_name,pair_of_pairs,'Slope'].values],
                                                    [0,NonBOLD_ideal_slope])]
        dist_NonBOLD_Y = dist_NonBOLD_Y + [euclidean([slope_inter_xr_all[y_label].loc[scan_name,pair_of_pairs,'Intercept'].values,
                                                     slope_inter_xr_all[y_label].loc[scan_name,pair_of_pairs,'Slope'].values],
                                                    [0,NonBOLD_ideal_slope])]
    df.loc[scan_name,'dist_BOLD_'+x_label] = np.array(dist_BOLD_X).mean()
    df.loc[scan_name,'dist_BOLD_'+y_label] = np.array(dist_BOLD_Y).mean()
    df.loc[scan_name,'dist_NonBOLD_'+x_label] = np.array(dist_NonBOLD_X).mean()
    df.loc[scan_name,'dist_NonBOLD_'+y_label] = np.array(dist_NonBOLD_Y).mean()
df.index.name='scan'

# # 7. Plotting Results
#
# Concatenate DBOLD / DnonBOLD and all other QA information we have about each scan into a single dataframe. This is convenient for them creating plots with the hvplot library

aux = pd.concat([df, mot_df, tedana_df,rsme_df],axis=1)
aux['Percent Censored'] = (aux['Percent Censored'].astype(float)+1)*2
aux['Percent Used'] = aux['Percent Used'].astype(float)
aux['Var. likely-BOLD'] = aux['Var. likely-BOLD'].astype(float)
aux['Var. unlikely-BOLD'] = aux['Var. unlikely-BOLD'].astype(float)
aux['Var. accepted'] = aux['Var. accepted'].astype(float)
aux['Var. rejected'] = aux['Var. rejected'].astype(float)
aux['Avg. RSME'] = aux['Avg. RSME'].astype(float)
aux.head(3)

aux.hvplot.scatter(x='dist_BOLD_'+x_label,   y='dist_BOLD_'+y_label,    aspect='square', cmap='viridis', 
                   c='Var. accepted', hover_cols=['scan'], alpha=0.7,s='Percent Censored', 
                   xlabel=r"\[x\pi\]",
                   xlim=(-.01,3.1), ylim=(-.01,3.1)).opts(clim=(0,30))* hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)+ \
aux.hvplot.scatter(x='dist_NonBOLD_'+x_label,y='dist_NonBOLD_'+y_label, aspect='square', cmap='viridis', c='Var. accepted', hover_cols=['scan'], alpha=0.7,s='Percent Censored', xlim=(-.01,3.1), ylim=(-.01,3.1)).opts(clim=(0,30)) * hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)

BOLD_plot = aux.hvplot.scatter(x='dist_BOLD_'+x_label,   y='dist_BOLD_'+y_label,    aspect='square', cmap='viridis', c='Var. accepted', 
                   hover_cols=['scan'],s='Percent Censored', alpha=0.7, xlim=(-.01,3.1), ylim=(-.01,3.1),
                   xlabel='D_BOLD for '+x_label, ylabel='D_BOLD for '+y_label,
                   fontsize={'ticks':8,'clabel':8,'xlabel':8,'ylabel':8}).opts(fontscale=1.5, clim=(0,30), colorbar_opts={'title':'% Var Accept. Components'})* \
            hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)

nonBOLD_plot = aux.hvplot.scatter(x='dist_NonBOLD_'+x_label,   y='dist_NonBOLD_'+y_label,    aspect='square', cmap='viridis', c='Var. accepted', 
                   hover_cols=['scan'],s='Percent Censored', alpha=0.7, xlim=(-.01,3.1), ylim=(-.01,3.1),
                   xlabel='D_nonBOLD for '+x_label, ylabel='D_nonBOLD for '+y_label,
                   fontsize={'ticks':8,'clabel':8,'xlabel':8,'ylabel':8}).opts(fontscale=1.5, clim=(0,30), colorbar_opts={'title':'% Var Accept. Components'})* \
            hv.Slope(1,0).opts(line_color='k', line_dash='dashed', line_width=0.5)

BOLD_plot

# +
top_double_pairs = ['e01|e01_vs_e02|e02','e01|e01_vs_e03|e03','e02|e02_vs_e03|e03'] #['e01|e01_vs_e03|e03'] #
dfs_to_plot = {}
oh = OpenHead(line_color='black', line_width=1)
nh = NormalHead(fill_color='black', fill_alpha=0.5, line_color='black',size=5)
intercept_range = Range1d(-0.1,0.2)
slope_range = Range1d(0,12)
layout = pn.layout.GridBox(ncols=3)

for pair_of_pairs in top_double_pairs:
    for label in [x_label,y_label]:
        aux_b = pd.DataFrame(slope_inter_xr_all[label].loc[:,pair_of_pairs,['Slope','Intercept']].values,
                           columns=['Slope','Intercept'],
                           index = list(slope_inter_xr_all[label].scan.values))
        aux_b.index.name = 'scan'
        aux_b = pd.concat([aux_b,aux[['Var. accepted','Percent Censored']]],axis=1)
        aux_b.index = [i+'|'+label for i in aux_b.index]
        plot = aux_b.hvplot.scatter(x='Intercept',y='Slope', aspect='square', hover_cols=['scan'], 
                                    title=label+' - '+pair_of_pairs, 
                                    cmap='viridis', c='Var. accepted',
                                    s='Percent Censored', alpha=0.7,
                                    ylim=(0,ideal_slopes[pair_of_pairs]*1.01)) * \
                hv.VLine(0).opts(line_width=1,line_color='g',line_dash='dashed') * \
                hv.HLine(ideal_slopes[pair_of_pairs]).opts(line_width=1,line_color='g',line_dash='dashed') * \
                hv.HLine(1).opts(line_width=1,line_color='r',line_dash='dashed')
        plot.opts(shared_axes=True, xlim=(-0.1,0.2), ylim=(0,12))
        layout.append(plot)
        dfs_to_plot[(label,pair_of_pairs)] = aux_b
    
    # Vector plot
    p_vectors = figure(tools=['pan','reset','box_zoom','wheel_zoom','save'], toolbar_location='right', 
                       background_fill_color="#ffffff", title=x_label+' --> '+y_label+' - '+pair_of_pairs)
    p_vectors.height=385
    p_vectors.width=500
    p_vectors.grid.grid_line_color = None

    approach = pd.DataFrame(np.sqrt((dfs_to_plot[x_label,pair_of_pairs]['Intercept'].values - 0)**2+(dfs_to_plot[x_label,pair_of_pairs]['Slope'].values - ideal_slopes[pair_of_pairs])**2) - \
                        np.sqrt((dfs_to_plot[y_label,pair_of_pairs]['Intercept'].values - 0)**2+(dfs_to_plot[y_label,pair_of_pairs]['Slope'].values - ideal_slopes[pair_of_pairs])**2),
                            index=scan_names,
                            columns=['approach'])
    approach.index.name = 'scan'
    approach_vmin = approach.quantile(0.05).values[0]
    approach_vmax = approach.quantile(0.95).values[0]
    if abs(approach_vmin) > abs(approach_vmin):
        approach_vmax = abs(approach_vmin)
    else:
        approach_vmin = -approach_vmax
    data = {'x':pd.concat([dfs_to_plot[(x_label,pair_of_pairs)].loc[:,'Intercept'],dfs_to_plot[(y_label,pair_of_pairs)].loc[:,'Intercept']],axis=0),
            'y':pd.concat([dfs_to_plot[(x_label,pair_of_pairs)].loc[:,'Slope'],dfs_to_plot[(y_label,pair_of_pairs)].loc[:,'Slope']],axis=0)}
    
    df = pd.DataFrame(data)#, index=aux_b.index)
    df.index.name = 'scan'

    # Prepare the data source
    df = df.reset_index()  # Reset index so 'scan' becomes a column
    source = ColumnDataSource(df)
    
    p_vectors.scatter(x='x', y='y', source=source, size=5, fill_color="blue", line_color="black", alpha=0)
    hover = HoverTool(tooltips=[("Scan", "@scan"), ("Intercept", "@x"), ("Slope", "@y")])
    p_vectors.add_tools(hover)

    for scan in list(approach.sort_values(by='approach',ascending=False).index):#scan_names:
        this_scan_approach = approach.loc[scan,'approach']
        approach_color = matplotlib_color_mapper(this_scan_approach, vmin=approach_vmin, vmax=approach_vmax)
        nh = NormalHead(fill_color=approach_color, fill_alpha=0.7, line_color=approach_color,size=5)
        x_start = dfs_to_plot[(x_label,pair_of_pairs)].loc[scan+'|'+x_label,'Intercept']
        y_start = dfs_to_plot[(x_label,pair_of_pairs)].loc[scan+'|'+x_label,'Slope']
        x_end   = dfs_to_plot[(y_label,pair_of_pairs)].loc[scan+'|'+y_label,'Intercept']
        y_end   = dfs_to_plot[(y_label,pair_of_pairs)].loc[scan+'|'+y_label,'Slope']
        p_vectors.add_layout(Arrow(end=nh, line_color=approach_color, line_width=1,
                                   x_start=x_start, y_start=y_start, 
                                   x_end=x_end, y_end=y_end, line_alpha=0.7))
    p_vectors.xaxis.axis_label = 'Intercept'
    p_vectors.yaxis.axis_label = 'Slope'
    p_vectors.x_range=intercept_range
    p_vectors.y_range=slope_range
    p_vectors.line([-0.1,0.2],[1,1],color='red', line_dash='dashed')
    p_vectors.line([-0.1,0.2],[ideal_slopes[pair_of_pairs],ideal_slopes[pair_of_pairs]],color='green', line_dash='dashed')
    p_vectors.line([0,0],[0,12],color='green', line_dash='dashed')
    # Create colorbar
    # Step 1: Define the matplotlib colormap and range
    mpl_cmap = plt.cm.seismic  # Replace with any matplotlib colormap
    norm = Normalize(vmin=approach_vmin, vmax=approach_vmax)  # Normalization range
    # Step 2: Convert the matplotlib colormap to a Bokeh palette
    # Convert RGBA to hex for Bokeh
    bokeh_palette = [rgb2hex(color) for color in mpl_to_bokeh_palette(mpl_cmap)]

    # Step 3: Create a Bokeh LinearColorMapper
    bokeh_color_mapper = LinearColorMapper(palette=bokeh_palette, low=approach_vmin, high=approach_vmax)

    # Step 4: Add colorbar
    color_bar = ColorBar(color_mapper=bokeh_color_mapper, label_standoff=12, location=(0, 0), title="DBOLD_"+x_label+" - DBOLD_"+y_label)
    p_vectors.add_layout(color_bar, 'right')
    layout.append(p_vectors)
# -

layout

layout.save('./Basic_2_MEICA.jpeg')

approach.quantile(0.05),approach.quantile(0.95)








