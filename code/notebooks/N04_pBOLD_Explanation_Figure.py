# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: BOLD WAVES 2024a
#     language: python
#     name: bold_waves_2024a
# ---

# # Description: Create figures to explain the logic behing pBOLD
#
# This notebook creates FC matrices and scatter plots for a representative subject. These figures are intended to help explain the goals of the project. They do not represent group-level results over which to draw final conclusions
#
# It will generate the plots both for R-FC (Pearson's Correlation based FC) and C-FC (Covariance based FC)

import pandas as pd
import numpy as np
import holoviews as hv
import os.path as osp
import os
from tqdm import tqdm
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, PRJ_DIR, CODE_DIR
from sfim_lib.plotting.fc_matrices import hvplot_fc
from itertools import combinations_with_replacement, combinations
ATLAS_NAME = 'Power264-discovery'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
import panel as pn
pn.extension()
from nilearn.connectome import sym_matrix_to_vec

port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# Sometimes, bokeh does not render properly in jupyter notebooks. The code on the following cell helps resolve this issue

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

# Create lists with all 6 possible echo combinations, and then all possible pairings between those.

echo_pairs_tuples   = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]
print('Echo Pairs[n=%d]=%s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d]=%s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

echo_times_dict = {'e01':13.9,'e02':31.7, 'e03':49.5}

# Pick a subjec and scan as representatives for the purpose of generating the figures

# Representative scan used when submitting the abstract to OHBM 2025
good_scan = ('MGSBJ07','constant_gated')
bad_scan  = ('MGSBJ07', 'cardiac_gated')
sample_scans = [good_scan, bad_scan, ('MGSBJ05','constant_gated'),('MGSBJ05','cardiac_gated'), 
                ('MGSBJ01','constant_gated'),('MGSBJ01','cardiac_gated'),
                ('MGSBJ02','constant_gated'),('MGSBJ02','cardiac_gated'),
                ('MGSBJ03','constant_gated'),('MGSBJ03','cardiac_gated'),
                ('MGSBJ04','constant_gated'),('MGSBJ04','cardiac_gated')]
sample_scans_select = {'constant gating':good_scan, 'cardiac gating':bad_scan}

# # 1. Load Atlas Information

# +
roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df.head(5)

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)

print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index
# -

# Create a dictionary to be used as colormap when plotting FC matrices

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

# # 2. Load Timeseries and compute R and C matrices
#
# This cell will load ROI Timeseries, compute R and C, and place these into a dictionary of datafrmes. It will do this for the Basic denoising pipeline (Basic) and no censoring (ALL).

scenarios             = ['ALL_Basic','ALL_GS','ALL_Tedana-fastica']
scenarios_select_dict = {'No Censoring | Basic':'ALL_Basic',
                         'No Censoring | GSR':'ALL_GS',
                         'No Censoring | tedana (fastica)':'ALL_Tedana-fastica'}
fc = {}

for (sbj,ses) in tqdm(sample_scans):
    for scenario in scenarios:
        for (e_x,e_y) in echo_pairs_tuples:
            roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{scenario}.{ATLAS_NAME}_000.netts')
            roi_ts_x      = np.loadtxt(roi_ts_path_x)
            roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{scenario}.{ATLAS_NAME}_000.netts')
            roi_ts_y      = np.loadtxt(roi_ts_path_y)
            aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
            aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
            # Compute the full correlation matrix between aux_ts_x and aux_ts_y
            aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
            aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
            fc['R',sbj, ses, scenario,(e_x,e_y)]  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)
            fc['C',sbj, ses, scenario,(e_x,e_y)]  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)

#

# ***
#
# # 3. Create Figure 1
#
# ## 3.1. Figure 1.a| Non-BOLD Data demonstration
#
# The cell below generates Figure 1.a where we show both a simulation and real case of how data that is non-BOLD dominated behaves.

# +
sbj='MGSBJ03'
ses='cardiac_gated'
zero_marker = hv.VLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray') * hv.HLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray')

a  = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Basic',('e01','e02')].values,discard_diagonal=True)
b  = a + 0.1 * (np.random.normal(0,.25,25425))

df = pd.DataFrame([a,b], index=['C(TE1,TE2)','C(TE1,TE3)']).T
plot_simulation = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='C(TEi,TEj)', ylabel='C(TEk,TEl)', xlim=(-.1,1.5), ylim=(-.1,1.5)).opts(fontscale=1.2, title='(a) Simulation', active_tools=['reset']) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=2) * zero_marker

c = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Basic',('e01','e03')].values,discard_diagonal=True)
df = pd.DataFrame([a,c], index=['C(TE1,TE2)','C(TE1,TE3)']).T
plot_real_data = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='C(TE1,TE2)', ylabel='C(TE1,TE3)', xlim=(-.1,1.5), ylim=(-.1,1.5)).opts(fontscale=1.2, title='(b) Real Data', active_tools=['reset']) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=2) * zero_marker
output = pn.Column(plot_simulation + plot_real_data)
output
# -

# ## 3.2. Figure 1.b| BOLD Data demonstration
#
# The cell below generates Figure 1.b where we show both a simulation and real case of how data that is BOLD-dominated behaves.

slope = (echo_times_dict['e02'] * echo_times_dict['e03']) / (echo_times_dict['e01'] * echo_times_dict['e02'])
slope

# +
sbj='MGSBJ03'
ses='constant_gated'
zero_marker = hv.VLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray') * hv.HLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray')

a  = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Tedana-fastica',('e01','e02')].values,discard_diagonal=True)
b  = (slope * a) + 0.1 * (np.random.normal(0,.25,25425))
df = pd.DataFrame([a,b], index=['C(TE1,TE2)','C(TE1,TE3)']).T
plot_simulation = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='C(TEi,TEj)', ylabel='C(TEk,TEl)', xlim=(-.6,.6), ylim=(-.6,.6)).opts(fontscale=1.2, title='(c) Simulation', active_tools=['reset']) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=2) * hv.Slope(slope,0).opts(line_color='g',line_dash='dashed',line_width=2) * zero_marker

c = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Tedana-fastica',('e02','e03')].values,discard_diagonal=True)
df = pd.DataFrame([a,c], index=['C(TE1,TE2)','C(TE2,TE3)']).T
plot_real_data = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE2,TE3)', aspect='square',color='black', datashade=True, xlabel='C(TE1,TE2)', ylabel='C(TE2,TE3)', xlim=(-.6,.6), ylim=(-.6,.6)).opts(fontscale=1.2, title='(d) Real Data', active_tools=['reset']) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=2) * hv.Slope(slope,0).opts(line_color='g',line_dash='dashed',line_width=2) *zero_marker
output = pn.Column(plot_simulation + plot_real_data)
output
# -

# # 4. Create Figure 2: How to Compute pBOLD step-by-step
# ## 4.1. Steps 1 - 3: Compute FCc for two different pairs of echo times and build a scatter plot

# +
from utils.basics import compute_residuals, project_points
from bokeh.models import FixedTicker

sbj_fig2       = 'MGSBJ05'
ses_fig2       = 'constant_gated'
pp_fig2        = 'ALL_Tedana-fastica'
fc_metric_fig2 = 'C'
x_te1,x_te2    = 'e01','e02'
y_te1,y_te2    = 'e02','e03'
# -

# Get Connectivity data: top triangle of the FC-C matrix
a = sym_matrix_to_vec(fc[fc_metric_fig2,sbj_fig2,ses_fig2,pp_fig2,(x_te1,x_te2)].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc[fc_metric_fig2,sbj_fig2,ses_fig2,pp_fig2,(y_te1,y_te2)].values,discard_diagonal=True)
df = pd.DataFrame([a,b], index=['C(TEi,TEj)','C(TEk,TEl)']).T
x = df['C(TEi,TEj)'].values
y = df['C(TEk,TEl)'].values

# Compute slopes of the two noise regimes of interest
So_slope   = 1.0
BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])

# +
# Create basic graphic elements: zero point, BOLD line, non-BOLD line, basic scatter, etc.
zero_point = hv.HLine(0).opts(line_width=0.5, line_color='k', line_dash='dotted') * hv.VLine(0).opts(line_width=0.5, line_color='k', line_dash='dotted')

#scat_datashaded = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), size=.75).opts(title='')
scat = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), rasterize=True, cmap='cividis', xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$",fontsize=14).opts(colorbar_opts={"title": "Num. Overlapping Edges"},toolbar='above', active_tools=['reset'])

nonBOLD_line = hv.Slope(1,0).opts(line_width=3, line_color='r', line_dash='dashed')
BOLD_line = hv.Slope(BOLD_slope,0).opts(line_width=3, line_color='g', line_dash='dashed') 

data_fit = np.polyfit(a,b,1)

data_line = hv.Slope(data_fit[0],data_fit[1]).opts(line_width=3, line_color='b', line_dash='dashed') 
print(data_fit)
# -

data        = fc[fc_metric_fig2 ,sbj_fig2,ses_fig2,pp_fig2,(x_te1,x_te2)]
plot_fc_x   = hvplot_fc(data,major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
         cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left',clim=(-.3,.3),
         cbar_title=r"$$FC_{C}(TE_{i},TE_{j})$$", cbar_title_fontsize=14, ticks_font_size=14).opts(tools=[])
data        = fc[fc_metric_fig2 ,sbj_fig2,ses_fig2,pp_fig2,(y_te1,y_te2)]
plot_fc_y   = hvplot_fc(data,major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
         cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left',clim=(-.3,.3),
         cbar_title=r"$$FC_{C}(TE_{k},TE_{l})$$", cbar_title_fontsize=14, ticks_font_size=14).opts(tools=[])
pn.Row(plot_fc_x,plot_fc_y)

(scat * zero_point)

# ## 4.2. Step 4: Estimate distance to line for BOLD dominated regime

# +
# Let's select a representative connection
# We pick on around 0.15, 0.25 becuase it is easy to visualize

x_target, y_target = 0.15,0.25

# Compute Euclidean distance
distances = np.sqrt((df["C(TEi,TEj)"] - x_target)**2 + (df["C(TEk,TEl)"] - y_target)**2)

# Get index of closest row
closest_idx = distances.idxmin()

# Extract the row
closest_row = df.loc[closest_idx]

sample_x,sample_y =  closest_row.values
sample_point = hv.Points((sample_x,sample_y)).opts(size=10,color='k', marker='x')
print(closest_row, sample_x, sample_y)
# -

proj_on_BOLD_line             = project_points(sample_x,sample_y,BOLD_slope,0.0)
proj_on_BOLD_line_point       = hv.Points(proj_on_BOLD_line).opts(size=10,color='g', marker='s')
proj_on_BOLD_residual_segment = hv.Path([[(sample_x,sample_y), proj_on_BOLD_line]]).opts(color="green", line_width=2)

(scat * zero_point) * BOLD_line * nonBOLD_line * sample_point * proj_on_BOLD_line_point * proj_on_BOLD_residual_segment

df['dBOLD'] = compute_residuals(df["C(TEi,TEj)"].values,df["C(TEk,TEl)"].values,BOLD_slope,0.0)

scat_colored_by_dBOLD = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), 
                                          size=.75, color='dBOLD', cmap='viridis',fontscale=1.3,
                                          xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$").opts(title='',clim=(df['dBOLD'].quantile(0.01),df['dBOLD'].quantile(0.95)),colorbar_opts={"title": "Distance to BOLD line"})

scat_colored_by_dBOLD * zero_point * BOLD_line

# ## 4.3. Step 5: Distance to line indicating So dominated data

proj_on_So_line             = project_points(sample_x,sample_y,So_slope,0.0)
proj_on_So_line_point       = hv.Points(proj_on_So_line).opts(size=10,color='r', marker='s')
proj_on_So_residual_segment = hv.Path([[(sample_x,sample_y), proj_on_So_line]]).opts(color="red", line_width=2)

(scat * zero_point) * BOLD_line * nonBOLD_line * sample_point * proj_on_So_line_point * proj_on_So_residual_segment

df['dSo'] = compute_residuals(df["C(TEi,TEj)"].values,df["C(TEk,TEl)"].values,1.0,0.0)

scat_colored_by_dSo = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), 
                                    size=.75, color='dSo', cmap='viridis',fontscale=1.3,
                                          xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$").opts(title='',
                                                                                 clim=(df['dSo'].quantile(0.01),df['dSo'].quantile(0.95)),
                                                                                colorbar_opts={"title": "Distance to So line"})

scat_colored_by_dSo * zero_point * nonBOLD_line

# ## 4.4. Step 6: Preference towards BOLD line

# Let's now see which line is the closest to each connection.
# When the difference in distance is below the tolerance, we call it a tie and assign a 0.5
pref1 = (df['dBOLD'] < df['dSo']).astype(float)
ties = np.isclose(df['dBOLD'], df['dSo'], atol=1e-3)
pref1[ties] = 0.5
df['prefBOLD'] = pref1

ticks = [0.25, 0.5, 0.75]
labels = {0.25: "non-BOLD", 0.5:"Uncertain", 0.75: "BOLD"}
scat_colored_by_prefBOLD = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), 
                                    size=.75, color='prefBOLD', cmap=['#FF7F7F','#FFFF7F','lightgreen'],xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$", fontscale=1.3).opts(title='',
                                                                                 clim=(df['dBOLD'].quantile(0.01),df['prefBOLD'].quantile(0.95)),
                                                                                colorbar_opts={"title": "Line preference (e.g., likely behavior):","ticker": FixedTicker(ticks=ticks),
                                                                                "major_label_overrides": labels,})

scat_colored_by_prefBOLD * zero_point * BOLD_line * nonBOLD_line

# ## 4.5. Step 7: Weigthed preference towards BOLD line

weight_fn     = lambda r: np.power(r,1.0)
max_weight_fn = lambda r: np.minimum(r,np.quantile(r,.95))

r  = np.sqrt(x**2 + y**2)

if weight_fn is None:
    weight_fn = lambda r: r   # linear weight by radius
w = weight_fn(r)
if max_weight_fn is not None:
    w = max_weight_fn(w)
total_weight = w.sum()

weighted_pref1 = (w * pref1)
frac_line1 = weighted_pref1

import matplotlib.colors as mcolors
df['w'] = w
df['w_prefBOLD'] = frac_line1
colors = ['#FF7F7F','#FFFF7F','lightgreen']
cmap = mcolors.LinearSegmentedColormap.from_list("green_yellow_red", colors)

scat_colored_by_w = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), 
                                    size=.75, color='w', cmap='gray_r',xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$", fontscale=1.3).opts(title='',
                                                                                 clim=(df['w'].quantile(0.05),df['w'].quantile(0.95)),
                                                                                colorbar_opts={"title": "Weights [Distance to origin]"})

scat_colored_by_w * zero_point * BOLD_line * nonBOLD_line

scat_colored_by_wprefBOLD = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), 
                                    size=.75, color='w_prefBOLD', cmap=cmap,xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$", fontscale=1.3).opts(title='',
                                                                                 clim=(df['w_prefBOLD'].quantile(0.05),df['w_prefBOLD'].quantile(0.95)),
                                                                                colorbar_opts={"title": "Weigthed Pref. towards BOLD line"})

scat_colored_by_wprefBOLD * zero_point * BOLD_line * nonBOLD_line

# ## 4.7 Chord Distance

# +
import holoviews as hv
import math
import math
from utils.basics import chord_distance_between_intersecting_lines

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


# +
# CASE 1:
x_te1,x_te2='e01','e03'
y_te1,y_te2='e03','e03'
BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])
BOLD_line  = hv.Slope(BOLD_slope,0).opts(line_width=3, line_color='g', line_dash='dashed') 

# Create an Ellipse element with center (0,0) and diameter 2 (radius 1)
unit_circle            = hv.Ellipse(0, 0, 1).opts(line_dash='dashed')
BOLD_point_on_circle   = np.array(line_circle_intersection(BOLD_slope,0.0,r=0.5))
So_point_on_circle     = np.array(line_circle_intersection(So_slope,0.0,r=0.5))
BOLD_dot               = hv.Points([BOLD_point_on_circle]).opts(size=5,color='g')
So_dot                 = hv.Points([So_point_on_circle]).opts(size=5,color='r')
BOLD2So_path           = hv.Path([[BOLD_point_on_circle, So_point_on_circle]]).opts(color="black", line_width=2)
segment                = np.sqrt((So_point_on_circle-BOLD_point_on_circle)**2)/2
BOLD2So_path_middle    = np.array([So_point_on_circle[0]-segment[0],So_point_on_circle[1]+segment[1]])
line_with_dist = hv.Arrow(BOLD2So_path_middle[0],BOLD2So_path_middle[1],'%.2f' % chord_distance_between_intersecting_lines(1.0, BOLD_slope, r=0.5),'^')

(scat.opts(colorbar=False) * BOLD_line * nonBOLD_line * zero_point * unit_circle * BOLD_dot * So_dot * BOLD2So_path * line_with_dist).opts(xlim=(-.51,.51), ylim=(-.51,.51), aspect='square', fontscale=1.3, 
                                                                                                                          xlabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[x_te1],echo_times_dict[x_te2]), 
                                                                                                                          ylabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[y_te1],echo_times_dict[y_te2]))

# +
# CASE 1:
x_te1,x_te2='e02','e02'
y_te1,y_te2='e02','e03'
BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])
BOLD_line  = hv.Slope(BOLD_slope,0).opts(line_width=3, line_color='g', line_dash='dashed') 

# Create an Ellipse element with center (0,0) and diameter 2 (radius 1)
unit_circle            = hv.Ellipse(0, 0, 1).opts(line_dash='dashed')
BOLD_point_on_circle   = np.array(line_circle_intersection(BOLD_slope,0.0,r=0.5))
So_point_on_circle     = np.array(line_circle_intersection(So_slope,0.0,r=0.5))
BOLD_dot               = hv.Points([BOLD_point_on_circle]).opts(size=5,color='g')
So_dot                 = hv.Points([So_point_on_circle]).opts(size=5,color='r')
BOLD2So_path           = hv.Path([[BOLD_point_on_circle, So_point_on_circle]]).opts(color="black", line_width=2)
segment                = np.sqrt((So_point_on_circle-BOLD_point_on_circle)**2)/2
BOLD2So_path_middle    = np.array([So_point_on_circle[0]-segment[0],So_point_on_circle[1]+segment[1]])
line_with_dist = hv.Arrow(BOLD2So_path_middle[0],BOLD2So_path_middle[1],'%.2f' % chord_distance_between_intersecting_lines(1.0, BOLD_slope, r=0.5),'^')

(scat.opts(colorbar=False) * BOLD_line * nonBOLD_line * zero_point * unit_circle * BOLD_dot * So_dot * BOLD2So_path * line_with_dist).opts(xlim=(-.51,.51), ylim=(-.51,.51), aspect='square', fontscale=1.3, 
                                                                                                                          xlabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[x_te1],echo_times_dict[x_te2]), 
                                                                                                                          ylabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[y_te1],echo_times_dict[y_te2]))
# -

# # 5. Supplementary Figure:  Demonstration of heteroscedasticity
#
# ## 5.1. SP X.a: Linear fit is not great

# +
x_te1,x_te2 = 'e01','e02'
y_te1,y_te2 = 'e02','e03'
a = sym_matrix_to_vec(fc['C','MGSBJ05','constant_gated','ALL_Tedana-fastica',(x_te1,x_te2)].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc['C','MGSBJ05','constant_gated','ALL_Tedana-fastica',(y_te1,y_te2)].values,discard_diagonal=True)
df = pd.DataFrame([a,b], index=['C(TE1,TE2)','C(TE2,TE3)']).T

zero_point = hv.HLine(0).opts(line_width=0.5, line_color='k', line_dash='dotted') * hv.VLine(0).opts(line_width=0.5, line_color='k', line_dash='dotted')

scat_datashaded = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE2,TE3)', aspect='square', xlim=(-.1,.5), ylim=(-.1,.5), datashade=True).opts(title='(a) Real Data', active_tools=['reset'])
scat = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE2,TE3)', aspect='square', xlim=(-.1,.5), ylim=(-.1,.5))

nonBOLD_line = hv.Slope(1,0).opts(line_width=3, line_color='r', line_dash='dashed')
BOLD_line = hv.Slope((echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2]),0).opts(line_width=3, line_color='g', line_dash='dashed') 

data_fit = np.polyfit(a,b,1)

data_line = hv.Slope(data_fit[0],data_fit[1]).opts(line_width=3, line_color='b', line_dash='dashed') 
print(data_fit)
# -

(scat_datashaded * zero_point) 

(scat_datashaded * nonBOLD_line * zero_point)

print((echo_times_dict['e02']*echo_times_dict['e03'])/(echo_times_dict['e01']*echo_times_dict['e02']))
(scat_datashaded * nonBOLD_line * BOLD_line * zero_point)

print(data_fit[0],data_fit[1])
(scat_datashaded * nonBOLD_line * BOLD_line * data_line * zero_point)

# ## 4.2 b| LOWESS Plot

# +
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag

X = a
# Add a constant for the intercept
X_const = sm.add_constant(X)
model = sm.OLS(b, X_const).fit()

# Run Breusch-Pagan test
# The function takes residuals and design matrix (exog)
test_stat, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)

print(f"Breusch-Pagan test statistic: {test_stat:.3f}")
print(f"P-value: {p_value:.3f}")

# -

df2=pd.DataFrame([model.fittedvalues,model.resid], index=['Fitted','Residuals']).T

# +
import pandas as pd
import holoviews as hv
import hvplot.pandas  # ensure hvplot is imported to extend pandas
from statsmodels.nonparametric.smoothers_lowess import lowess

hv.extension('bokeh')

# Compute LOWESS smoother
smoothed = lowess(df2['Residuals'], df2['Fitted'], frac=0.3)
smooth_df = pd.DataFrame(smoothed, columns=['Fitted', 'Residuals'])

# Scatter plot of residuals
scatter = df2.hvplot.scatter(x='Fitted', y='Residuals', alpha=0.7, size=3, color='k', datashade=True, ylim=(-.1,.2), xlim=(-.1,.2)).opts(active_tools=['reset'])

# Smoothed curve
smooth_curve = smooth_df.hvplot.line(x='Fitted', y='Residuals', color='red', name='LOWESS Fit')

# Horizontal zero line
zero_line = hv.HLine(0).opts(color='black', line_dash='dashed', line_width=0.5)

# Combine plots
(scatter * smooth_curve * zero_line).opts(
    title='(b) Fitted Values vs. Residuals & LOWESS line',
    width=600, height=400,
    xlabel='Fitted values', ylabel='Residuals')
# -
# # 6. Draw an example of FC following two different denosing methods and the difficulty with deciding which one is best
#
# Below we show the full brain R-FC matrices of a single scan for the basic (left) and tedana (right) pipelines. Just by looking at then, it not easy to discern which of these two matrices is a more truthful representation of neurally-driven connectivity.

scan_select      = pn.widgets.Select(name='Sample scan', options=sample_scans_select, width=200)
scenarioA_select = pn.widgets.Select(name='Left Configuration', options=scenarios_select_dict, width=200)
scenarioB_select = pn.widgets.Select(name='Right Configuration', options=scenarios_select_dict, width=200)
conf_card        = pn.Card(scan_select,scenarioA_select,scenarioB_select, title='Configuration')


def plot_matrix(scan,scenario,fc_metric='C',echo_pair=('e01','e02'), title=''):
    sbj_id = scan[0]
    run_id = scan[1]
    data   = fc[fc_metric,sbj_id,run_id,scenario,echo_pair]
    plot   = hvplot_fc(data,major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
                       cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', 
                       cbar_title=f"FC-{fc_metric}", cbar_title_fontsize=14, ticks_font_size=14).opts(tools=[],title=title)
    return plot
@pn.depends(scan_select,scenarioA_select)
def plot_left_matrix(scan,scenario):
    return plot_matrix(scan,scenario, title='Scenario B: '+ scenario)
@pn.depends(scan_select,scenarioB_select)
def plot_right_matrix(scan,scenario):
    return plot_matrix(scan,scenario, title='Scenario A:' + scenario)   


dashboard = pn.Row(conf_card,plot_left_matrix, plot_right_matrix)

dashboard_server = dashboard.show(port=port_tunnel)

dashboard.stop()
