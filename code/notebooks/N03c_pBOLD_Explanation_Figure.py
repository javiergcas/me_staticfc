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
ATLAS_NAME = 'Power264_GatingDataset'
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
sample_scans = [good_scan, bad_scan, ('MGSBJ05','constant_gated')]
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

scenarios             = ['ALL_Basic','ALL_GSasis','ALL_Tedana']
scenarios_select_dict = {'No Censoring | Basic':'ALL_Basic','No Censoring | GSR':'ALL_GSasis','No Censoring | tedana':'ALL_Tedana'}
fc = {}

for (sbj,ses) in sample_scans:
    for scenario in scenarios:
        for (e_x,e_y) in tqdm(echo_pairs_tuples, desc=scenario):
            roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
            roi_ts_x      = np.loadtxt(roi_ts_path_x)
            roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
            roi_ts_y      = np.loadtxt(roi_ts_path_y)
            aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
            aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
            # Compute the full correlation matrix between aux_ts_x and aux_ts_y
            aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
            aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
            fc['R',sbj, ses, scenario,(e_x,e_y)]  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)
            fc['C',sbj, ses, scenario,(e_x,e_y)]  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)

# # 3. Draw an example of FC following two different denosing methods and the difficulty with deciding which one is best
#
# Below we show the full brain R-FC matrices of a single scan for the basic (left) and tedana (right) pipelines. Just by looking at then, it not easy to discern which of these two matrices is a more truthful representation of neurally-driven connectivity.

scan_select      = pn.widgets.Select(name='Sample scan', options=sample_scans_select, width=200)
scenarioA_select = pn.widgets.Select(name='Left Configuration', options=scenarios_select_dict, width=200)
scenarioB_select = pn.widgets.Select(name='Right Configuration', options=scenarios_select_dict, width=200)
conf_card        = pn.Card(scan_select,scenarioA_select,scenarioB_select, title='Configuration')


def plot_matrix(scan,scenario,fc_metric='R',echo_pair=('e02','e02'), title=''):
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

dashboard_server.stop()

# ***

hvplot_fc(fc['R',bad_scan[0],bad_scan[1],'ALL_Basic',('e02','e02')],
          major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', cbar_title="Pearson's Correlation:",cbar_title_fontsize=14,ticks_font_size=14).opts(default_tools=["pan"]).opts(title='Cardiac Gated | Basic Denoising | No Censoring') + \
hvplot_fc(fc['R',bad_scan[0],bad_scan[1],'ALL_Tedana',('e02','e02')],
          major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', cbar_title="Pearson's Correlation:",cbar_title_fontsize=14,ticks_font_size=14).opts(default_tools=["pan"]).opts(title='Cardiac Gated | Tedana Denoising | No Censoring')

# ***
# # 4. How does FC-R behave across echoes?
#
# ## 4.1. General Behavior with no approximations
# Let's describe the mono-exponential decay signal at two locations (x and y) for two different echo times (TEi and TEj) as follows:
# $$s_{x,i}=s(x,t,TE_i)=\Delta\rho(x,t)-\Delta R_2^*(x,t)\cdot TE_i + \epsilon(x,t) = \Delta\rho_{x} - {\Delta R_2^*}_{x} \cdot TE_i + \epsilon_{x}\tag{1}$$
#
# $$s_{y,j}=s(y,t,TE_j)=\Delta\rho(y,t)-\Delta R_2^*(y,t)\cdot TE_j + \epsilon(y,t)= \Delta\rho_{y} - {\Delta R_2^*}_{y} \cdot TE_i + \epsilon_{y}\tag{2}$$
#
# where $\Delta\rho$ denotes fluctuations in net magnetization, $\Delta R_2^*$ fluctuations of BOLD origin and $\epsilon$ is the thermal noise.
#
# Moreover, functional connectivity between two locations $x$ and $y$ is often estimated in terms of Pearson's Correlation Eq $(3)$ as follows:
#
# $$R(s_{x,i},s_{y,j}) = R_{xi,yj} = \frac{\sum{(s_{x,i}-\overline{s_{x,i}}) \cdot (s_{y,j}-\overline{s_{y,j}})}} {\sqrt{ \sum{(s_{x,i}-\overline{s_{x,i}})^2} \cdot \sum{(s_{y,i}-\overline{s_{x,i}})^2} }} \tag{3}$$
#
# Becuase $s$ here always represents signals in units of signal percent change, its mean is zero (e.g., $\overline{s_{x,i}}=\overline{s_{y,j}}=0$), and therefore Eq. $(3)$ can be simplified as follows:
#
# $$R_{xi,yj} = \frac{\sum{s_{x,i} \cdot s_{y,j}}} {\sqrt{ \sum{s_{x,i}^2} \cdot \sum{s_{y,j}^2} }} \tag{4}$$
#
# Under no additional assumptions, we have the following formula, which is hard to simplify once we expand the multiplicative terms
#
#
# $$R_{xi,yj} = \frac{\sum{(\Delta\rho_{x} - {\Delta R_2^*}_{x} \cdot TE_i + \epsilon_{x}) \cdot (\Delta\rho_{y} - {\Delta R_2^*}_{y} \cdot TE_j + \epsilon_{y})}} {\sqrt{ \sum{(\Delta\rho_{x} - {\Delta R_2^*}_{x} \cdot TE_i+ \epsilon_{x})^2} \cdot \sum{(\Delta\rho_{y} - {\Delta R_2^*}_{y} \cdot TE_j+ \epsilon_{y})^2} }} \tag{5}$$
#
#
# ## 4.2 BOLD Fluctuations dominate over everything else
#
# Mathematically, this can be expresses as follows:
#
# $${\Delta R_2^*} \cdot TE >> \Delta\rho + \epsilon \tag{6}$$
#
# Consequently, the signal equation can be simplified as follows:
#
# $$s_{x,i}\approx- {\Delta R_2^*}_{x} \cdot TE_i \land s_{y,j}\approx- {\Delta R_2^*}_{y} \cdot TE_j \tag{7}$$
#
# If we introduce this simplified formulation of the recorded signals, we can observe the following
#
# $$R_{xi,yj} \approx \frac{\sum{(- {\Delta R_2^*}_{x}\cdot TE_i) \cdot (- {\Delta R_2^*}_{y}\cdot TE_j)}} {\sqrt{ \sum{(- {\Delta R_2^*}_{x} \cdot TE_i)^2} \cdot \sum{(- {\Delta R_2^*}_{y} \cdot TE_j)^2} }}= \frac{(TE_i \cdot TE_j) \sum{({\Delta R_2^*}_{x} \cdot {\Delta R_2^*}_{y})}} {(TE_i \cdot TE_j) \cdot \sqrt{ \sum{(- {\Delta R_2^*}_{x})^2} \cdot \sum{(- {\Delta R_2^*}_{y})^2} }} =  \frac{\sum{({\Delta R_2^*}_{x} \cdot {\Delta R_2^*}_{y})}} {\sqrt{ \sum{(- {\Delta R_2^*}_{x})^2} \cdot \sum{(- {\Delta R_2^*}_{y})^2} }} \tag{8}$$
#
# <center><u><b>FC-R in this scenario is TE-independent</b></u></center>
#
# ## 4.3 Non-BOLD Fluctuations dominate over everything else
#
# In this scenario, we are assuming that
#
# $${\Delta R_2^*} \cdot TE << \Delta\rho + \epsilon \tag{9}$$
#
# And therefore, the signal at a given location and echo time can be simplified as:
#
# $$s_{x,i} \approx  \Delta\rho_{x} + \epsilon_{x}\tag{10}$$
#
# Moreover, FC-R in this case reduces to 
#
# $$R_{xi,yj} \approx \frac{\sum{(\Delta\rho_{x} + \epsilon_{x})\cdot (\Delta\rho_{y} + \epsilon_{y})}}{\sqrt{\sum{(\Delta\rho_{x} + \epsilon_{x})^2}}\cdot\sqrt{\sum{(\Delta\rho_{y} + \epsilon_{y})^2}}}$$
#
# Nothing in the above formula depends on echo time, and therfore:
#
# <center><u><b>FC-R in this scenario is TE-independent</b></u></center>
# <br>
# <center><h4>In summary, FC-R is TE independent when the data is dominated by a single form of fluctuation, no matter whether that is BOLD or non-BOLD fluctuations</h4></center>

# +
zero_marker = hv.VLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray') * hv.HLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray')

np.random.seed(50)
sample_pairs = np.random.choice(echo_pairs,2,False)
print(sample_pairs)
plots = pn.Row()
for (sbj,ses),scenario, label in zip([good_scan,bad_scan],['ALL_Tedana','ALL_Basic'],['Const. Gated,Tedana -> BOLD Dominated','Cardiac Gated,Basic -> Non-BOLD Dominated']):
    aux_fc_x = sym_matrix_to_vec(fc['R',sbj, ses, scenario,tuple(sample_pairs[0].split('|'))].values,discard_diagonal=True)
    aux_fc_y = sym_matrix_to_vec(fc['R',sbj, ses, scenario,tuple(sample_pairs[1].split('|'))].values,discard_diagonal=True)
    e1_X,e2_X = sample_pairs[0].split('|')
    e1_Y,e2_Y = sample_pairs[1].split('|')
    
    df = pd.DataFrame([aux_fc_x,aux_fc_y], index=['FC-R (%s,%s)' % tuple(sample_pairs[0].split('|')),'FC-R (%s,%s)' % tuple(sample_pairs[1].split('|'))]).T

    plot = df.hvplot.hexbin(x=df.columns[0], y=df.columns[1], aspect='square',
                         color='black').opts(fontscale=1, title=label, xlim=(-.5,1),ylim=(-.5,1)) * \
           hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=2) * \
           zero_marker
    plots.append(plot)
# -

plots

# # 5. How does FC-C behave across echoes?
# ## 5.1. General Behavior with no approximations
#
# If one estimates FC using covariance (instead of correlation), then FC is expressed as follows:
#
# $$C(s_{x,i},s_{y,j}) = C_{xi,yj} = \frac{1}{N_{t}} \cdot \sum{(s_{x,i}-\overline{s_{x,i}}) \cdot (s_{y,j}-\overline{s_{y,j}})} \tag{11}$$
#
# Again, becuase $s$ here always represents signals in units of signal percent change, its mean is zero (e.g., $\overline{s_{x,i}}=\overline{s_{y,j}}=0$), and therefore Eq. $(11)$ can be simplified as follows:
#
# $$C_{xi,yj} = \frac{1}{N_{t}} \cdot \sum{(s_{x,i} \cdot s_{y,j})} = \frac{1}{N_{t}} \cdot \sum{(\Delta\rho_{x} - {\Delta R_2^*}_{x} \cdot TE_i + \epsilon_{x}) \cdot (\Delta\rho_{y} - {\Delta R_2^*}_{y} \cdot TE_j + \epsilon_{x})}  \tag{12}$$
#
# ## 5.2 BOLD Fluctuations dominate over everything else
#
# Mathematically, this can be expresses as follows:
#
# $${\Delta R_2^*} \cdot TE >> \Delta\rho + \epsilon \tag{13}$$
#
# Consequently, the signal equation can be simplified as follows:
#
# $$s_{x,i}\approx- {\Delta R_2^*}_{x} \cdot TE_i \land s_{y,j}\approx- {\Delta R_2^*}_{y} \cdot TE_j \tag{14}$$
#
# If we introduce this simplified formulation of the recorded signals, we can observe the following
#
# $$C_{xi,yj} \approx \frac{1}{N_{t}} \cdot \sum{(- {\Delta R_2^*}_{x} \cdot TE_i) \cdot (- {\Delta R_2^*}_{y} \cdot TE_j)} = TE_i \cdot TE_j\cdot \frac{1}{N_t}\sum{{\Delta R_2^*}_{x} \cdot {\Delta R_2^*}_{y}} \tag{15}$$
#
# <center><b><u>And therefore FC-C in this scenario is expected to be echo time dependent</u></b></center>
#
# > This means that now if we plot FC-C computed with one pair of echo times against FC-C computed using a different set of echo times, we should not expect it to fall over the identity line (black line below), but over a line with zero intercept and a slope proportional to the ratio of the contributing echo times. This is exemplified in the following figure.
#
# ## 5.3 Non-BOLD Fluctuations dominate over everything else
#
# In this scenario, we are assuming that
#
# $${\Delta R_2^*} \cdot TE << \Delta\rho + \epsilon \tag{9}$$
#
# And therefore, the signal at a given location and echo time can be simplified as:
#
# $$s_{x,i} \approx  \Delta\rho_{x} + \epsilon_{x}\tag{10}$$
#
# Moreover, FC-C in this case reduces to:
#
# $$C_{xi,yj} = \frac{1}{N_{t}} \cdot \sum{(s_{x,i} \cdot s_{y,j})} = \frac{1}{N_{t}} \cdot \sum{(\Delta\rho_{x} + \epsilon_{x}) \cdot (\Delta\rho_{y} + \epsilon_{x})}  \tag{16}$$
#
# Nothing in the above formula depends on echo time, and therfore:
#
# <center><u><b>FC-R in this scenario is TE-independent</b></u></center>
# <br>
# <center><h4>In summary, FC-C is TE dependent when the data is dominated BOLD fluctuations and TE independent otherwise</h4></center>

# +
zero_marker = hv.VLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray') * hv.HLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray')

np.random.seed(51)
sample_pairs = np.random.choice(echo_pairs,2,False)
print(sample_pairs)
plots = pn.Row()
for (sbj,ses),scenario, label in zip([good_scan,bad_scan],['ALL_Tedana','ALL_Basic'],['Const. Gated,Tedana -> BOLD Dominated','Cardiac Gated,Basic -> Non-BOLD Dominated']):
    aux_fc_x = sym_matrix_to_vec(fc['C',sbj, ses, scenario,tuple(sample_pairs[0].split('|'))].values,discard_diagonal=True)
    aux_fc_y = sym_matrix_to_vec(fc['C',sbj, ses, scenario,tuple(sample_pairs[1].split('|'))].values,discard_diagonal=True)
    # Allow line to not go thorugh zero
    #emp_slope, emp_intercept = np.polyfit(aux_fc_x,aux_fc_y,deg=1)
    # Force line to go throught zero
    emp_slope = np.dot(aux_fc_x, aux_fc_y) / np.dot(aux_fc_x, aux_fc_x)
    emp_intercept= 0
    
    e1_X,e2_X = sample_pairs[0].split('|')
    e1_Y,e2_Y = sample_pairs[1].split('|')
    th_slope  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])

    df = pd.DataFrame([aux_fc_x,aux_fc_y], index=['FC-C (%s,%s)' % tuple(sample_pairs[0].split('|')),'FC-C (%s,%s)' % tuple(sample_pairs[1].split('|'))]).T

    plot = df.hvplot.hexbin(x=df.columns[0], y=df.columns[1], aspect='square',
                         color='black').opts(fontscale=1, title=label, xlim=(-1,3),ylim=(-1,3)) * \
           hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=2) * \
           hv.Slope(th_slope,0).opts(line_color='g',line_dash='dashed',line_width=2) * zero_marker * \
           hv.Slope(emp_slope,emp_intercept).opts(line_color='b',line_dash='dashed',line_width=2) 

    plots.append(plot)
plots
# -

# ***

# +
zero_marker = hv.VLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray') * hv.HLine(0).opts(line_width=0.5, line_dash='dashed', line_color='gray')
a = sym_matrix_to_vec(fc['R',sbj,ses,'ALL_Tedana',('e01','e02')].values,discard_diagonal=True)
b = a + 0.1 * (np.random.rand(25425) - .5)
df = pd.DataFrame([a,b], index=['FC-R (TE1,TE2)','FC-R (TE1,TE3)']).T
plot_simulation = df.hvplot.scatter(x='FC-R (TE1,TE2)',y='FC-R (TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-R (TEi,TEj)', ylabel='FC-R (TEk,TEl)').opts(fontscale=1.5, title='Simulated behavior for FC-R') * hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2) * zero_marker

c = sym_matrix_to_vec(fc['R',sbj,ses,'ALL_Tedana',('e01','e03')].values,discard_diagonal=True)
df = pd.DataFrame([a,c], index=['FC-R (TE1,TE2)','FC-R (TE1,TE3)']).T
plot_real_data = df.hvplot.scatter(x='FC-R (TE1,TE2)',y='FC-R (TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-R (TE1,TE2)', ylabel='FC-R (TE1,TE3)').opts(fontscale=1.5, title='Empirical data') * hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2) * zero_marker

header = pn.pane.Markdown("""
# Exmaple of how FC-R is expected to behave across echoes. \n

Theory says that independenly of what noise source dominates the data, FC-R should be echo time independent. The simulation on the left,
and the represenative data on the right shows this behavior in the form of a scatter plot of FC-R between two different echo combinations.
As FC-R is expected to be echo independent, data sits approximately on the identity line (Slope=1, Intercept=0) plot_simulation + plot_real_data""", width=800)

output = pn.Column(header,plot_simulation + plot_real_data)
#output.save('../../results/FCR_theoretical_behavior_across_echoes.html')
output

# +
a = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Tedana',('e01','e02')].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Tedana',('e02','e03')].values,discard_diagonal=True)
theoretical_slope= (echo_times_dict['e02']*echo_times_dict['e03'])/(echo_times_dict['e01']*echo_times_dict['e02'])
df = pd.DataFrame([a,b], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
empirical_slope, empirical_intercept = np.polyfit(df[df.columns[0]],df[df.columns[1]],deg=1)
plot_simulation = df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-C (TE1,TE2)', ylabel='FC-C (TE2,TE3)').opts(fontscale=1.5, title='First set of TE pairs', xlim=(-.1,.3), ylim=(-.1,0.3)) * \
                  hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2)  * \
                  hv.Slope(theoretical_slope,0).opts(line_color='g',line_dash='dashed',line_width=2) * \
                  hv.Slope(empirical_slope,empirical_intercept).opts(line_color='b',line_dash='dashed',line_width=2) * \
                  zero_marker

c = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_Tedana',('e01','e03')].values,discard_diagonal=True)
theoretical_slope= (echo_times_dict['e01']*echo_times_dict['e03'])/(echo_times_dict['e01']*echo_times_dict['e02'])
df = pd.DataFrame([a,c], index=['FC-C (TE1,TE2)','FC-C (TE1,TE3)']).T
empirical_slope, empirical_intercept = np.polyfit(df[df.columns[0]],df[df.columns[1]],deg=1)
plot_real_data = df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-C (TE1,TE2)', ylabel='FC-C (TE1,TE3)').opts(fontscale=1.5, title='Second set of TE pairs', xlim=(-.1,.3), ylim=(-.1,0.3)) * \
                 hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2) * \
                 hv.Slope(theoretical_slope,0).opts(line_color='g',line_dash='dashed',line_width=2) * \
                 hv.Slope(empirical_slope,empirical_intercept).opts(line_color='b',line_dash='dashed',line_width=2) * \
                 zero_marker

header = pn.pane.Markdown("""
# Exmaple of how FC-C is expected to behave across echoes. \n
""")
output = pn.Column(header,plot_simulation + plot_real_data)
output.save('../../results/FCC_theoretical_behavior_across_echoes.html')
output

# +
a = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_GSasis',('e01','e02')].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_GSasis',('e02','e03')].values,discard_diagonal=True)
theoretical_slope= (echo_times_dict['e02']*echo_times_dict['e03'])/(echo_times_dict['e01']*echo_times_dict['e02'])
df = pd.DataFrame([a,b], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
empirical_slope, empirical_intercept = np.polyfit(df[df.columns[0]],df[df.columns[1]],deg=1)

plot_simulation = df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-C (TE1,TE2)', ylabel='FC-C (TE2,TE3)').opts(fontscale=1.5, title='First set of TE pairs', xlim=(-.1,.3), ylim=(-.1,0.3)) * \
                  hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2) * \
                  hv.Slope(theoretical_slope,0).opts(line_color='g',line_dash='dashed',line_width=2) * \
                  hv.Slope(empirical_slope,empirical_intercept).opts(line_color='b',line_dash='dashed',line_width=2) * \
                  zero_marker

c = sym_matrix_to_vec(fc['C',sbj,ses,'ALL_GSasis',('e01','e03')].values,discard_diagonal=True)
theoretical_slope= (echo_times_dict['e01']*echo_times_dict['e03'])/(echo_times_dict['e01']*echo_times_dict['e02'])
df = pd.DataFrame([a,c], index=['FC-C (TE1,TE2)','FC-C (TE1,TE3)']).T
empirical_slope, empirical_intercept = np.polyfit(df[df.columns[0]],df[df.columns[1]],deg=1)

plot_real_data = df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE1,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-C (TE1,TE2)', ylabel='FC-C (TE1,TE3)').opts(fontscale=1.5, title='Second set of TE pairs', xlim=(-.1,.3), ylim=(-.1,0.3)) * \
                 hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2) * \
                 hv.Slope(theoretical_slope,0).opts(line_color='g',line_dash='dashed',line_width=2) * \
                 hv.Slope(empirical_slope,empirical_intercept).opts(line_color='b',line_dash='dashed',line_width=2) * \
                 zero_marker

header = pn.pane.Markdown("""
# Exmaple of how FC-C is expected to behave across echoes. \n
""")
output = pn.Column(header,plot_simulation + plot_real_data)
output.save('../../results/FCC_theoretical_behavior_across_echoes.html')
output
# -

# ***

a = sym_matrix_to_vec(fc['R',sbj,ses,'ALL_Tedana',('e02','e02')].values,discard_diagonal=True)
b = a + 0.1 * (np.random.rand(25425) - .5)
df = pd.DataFrame([a,b], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
c = 2.3*a + 0.1 * (np.random.rand(25425) - .5)
df2 = pd.DataFrame([a,c], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True, xlabel='FC-C (TEi,TEj)', ylabel='FC-C (TEk,TEl)').opts(fontscale=1.5) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=3) * \
df2.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True).opts(fontscale=1.5) * hv.Slope(2.3,0).opts(line_color='g',line_dash='dashed',line_width=3)

a = sym_matrix_to_vec(fc['R',sbj,ses,'ALL_Tedana',('e02','e02')].values,discard_diagonal=True)
b = a + 0.1 * (np.random.rand(25425) - .5)
df = pd.DataFrame([a,b], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
c = 2.3*a + 0.1 * (np.random.rand(25425) - .5)
df2 = pd.DataFrame([a,c], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
pn.Row(df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True, xlabel='FC-C (TEi,TEj)', ylabel='FC-C (TEk,TEl)', xlim=(-.4,1.6), ylim=(-.4,1.6)).opts(fontscale=1.5) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=3),
df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True, xlabel='FC-C (TEi,TEj)', ylabel='FC-C (TEk,TEl)', xlim=(-.4,1.6), ylim=(-.4,1.6)).opts(fontscale=1.5) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=3) * \
df2.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True).opts(fontscale=1.5) * hv.Slope(2.3,0).opts(line_color='g',line_dash='dashed',line_width=3))

# +
import numpy as np

def variance_explained(x, y, degree):
    """
    Calculates the variance explained (R^2) by a polynomial fit.

    Args:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data.
        degree (int): Degree of the polynomial to fit.

    Returns:
        float: The variance explained (R^2), ranging from 0 to 1.
               Values closer to 1 indicate a better fit.
    """
    # Fit the polynomial
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    # Calculate the total sum of squares (SST)
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean)**2)

    # Calculate the sum of squared residuals (SSR)
    ssr = np.sum((y - p(x))**2)

    # Calculate R-squared
    r_squared = 1 - (ssr / sst)

    return r_squared,coeffs
# -


