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
# This notebook creates the FC and scatter plots for a representative subject.
#
# It will generate the plots both for R-FC and C-FC

import pandas as pd
import numpy as np
import holoviews as hv
import os.path as osp
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, ATLAS_NAME, PRJ_DIR, CODE_DIR
from sfim_lib.plotting.fc_matrices import hvplot_fc
from itertools import combinations_with_replacement, combinations
ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
import panel as pn
pn.extension()
from nilearn.connectome import sym_matrix_to_vec

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

echo_pairs_tuples   = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]
print('Echo Pairs[n=%d]=%s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d]=%s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

# Representative scan used when submitting the abstract to OHBM 2025
#sbj='sub-156'
#ses='ses-2'
sbj='sub-211'
ses='ses-2'

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

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

# # 2. Load Timeseries and compute R and C matrices
#
# This will work with the data following Basic denoising

#scenario = 'ALL_GSasis'
scenario = 'ALL'
fc = {}
for (e_x,e_y) in echo_pairs_tuples:
    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_x      = np.loadtxt(roi_ts_path_x)
    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_y      = np.loadtxt(roi_ts_path_y)
    aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
    aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
    # Compute the full correlation matrix between aux_ts_x and aux_ts_y
    aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    fc['R','Basic',(e_x,e_y)]  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)
    fc['C','Basic',(e_x,e_y)]  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)

for (e_x,e_y) in echo_pairs_tuples:
    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.meica_dn.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_x      = np.loadtxt(roi_ts_path_x)
    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.meica_dn.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_y      = np.loadtxt(roi_ts_path_y)
    aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
    aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
    # Compute the full correlation matrix between aux_ts_x and aux_ts_y
    aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    fc['R','Advanced',(e_x,e_y)]  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)
    fc['C','Advanced',(e_x,e_y)]  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)

# # 3. Draw an example of FC following two different denosing methods and the difficulty with deciding which one is best

hvplot_fc(fc['R','Basic',('e02','e02')],
          major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', cbar_title="Pearson's Correlation:",cbar_title_fontsize=14,ticks_font_size=14).opts(default_tools=["pan"]) + \
hvplot_fc(fc['R','Advanced',('e02','e02')],
          major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', cbar_title="Pearson's Correlation:",cbar_title_fontsize=14,ticks_font_size=14).opts(default_tools=["pan"])

# # 4. Plot of scatter for ideal scenarios in FC-R

a = sym_matrix_to_vec(fc['R','Advanced',('e02','e02')].values,discard_diagonal=True)
b = a + 0.1 * (np.random.rand(20503) - .5)
df = pd.DataFrame([a,b], index=['FC-R (TE1,TE2)','FC-R (TE2,TE3)']).T
df.hvplot.scatter(x='FC-R (TE1,TE2)',y='FC-R (TE2,TE3)', aspect='square',color='black', datashade=True, xlabel='FC-R (TEi,TEj)', ylabel='FC-R (TEk,TEl)').opts(fontscale=1.5) * hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=2)

# # 5. Plot of scatter for ideal scenarios in FC-C

a = sym_matrix_to_vec(fc['R','Advanced',('e02','e02')].values,discard_diagonal=True)
b = a + 0.1 * (np.random.rand(20503) - .5)
df = pd.DataFrame([a,b], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
c = 2.3*a + 0.1 * (np.random.rand(20503) - .5)
df2 = pd.DataFrame([a,c], index=['FC-C (TE1,TE2)','FC-C (TE2,TE3)']).T
df.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True, xlabel='FC-C (TEi,TEj)', ylabel='FC-C (TEk,TEl)').opts(fontscale=1.5) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=3) * \
df2.hvplot.scatter(x='FC-C (TE1,TE2)',y='FC-C (TE2,TE3)', aspect='square',c='r',s=1, datashade=True).opts(fontscale=1.5) * hv.Slope(2.3,0).opts(line_color='g',line_dash='dashed',line_width=3)

# # 6. Plot used to describe de DBOLD metric in the abstract

echoes_dict = {'e01':13.7,'e02':30,'e03':47}

a = sym_matrix_to_vec(fc['C','Advanced',('e01','e01')].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc['C','Advanced',('e02','e02')].values,discard_diagonal=True)
slope,intercept = np.polyfit(a,b,1)
df = pd.DataFrame([a,b], index=['FC-C(TEi,TEj)','FC-C (TEk,TEl)']).T
exp_slope = (echoes_dict['e02']**2)/(echoes_dict['e01']*echoes_dict['e01'])
print(exp_slope)
df.hvplot.scatter(x=df.columns[0],y='FC-C (TEk,TEl)', aspect='square',c='r',s=1, datashade=True).opts(fontscale=1.5) * \
hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=4) * \
hv.Slope(exp_slope,0).opts(line_color='g',line_dash='dashed',line_width=4) * \
hv.Slope(slope,intercept).opts(line_color='b',line_dash='dashed',line_width=4) 

# # 7. R-based Results
#
# ## 7.1 FC matrices for all 9 echo pairs
#
# Only two of these are used in Figure 2.A

sbj='sub-156'
ses='ses-2'

fc = {}
for (e_x,e_y) in echo_pairs_tuples:
    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_x      = np.loadtxt(roi_ts_path_x)
    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_y      = np.loadtxt(roi_ts_path_y)
    aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
    aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
    # Compute the full correlation matrix between aux_ts_x and aux_ts_y
    aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    fc['R','Basic',(e_x,e_y)]  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)
    fc['C','Basic',(e_x,e_y)]  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)

fc_plot_R = pn.FlexBox()
for (e_x,e_y) in echo_pairs_tuples:
    aux_fc = fc['R','Basic',(e_x,e_y)]
    fc_plot_R.append(hvplot_fc(aux_fc,
          major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', cbar_title="Pearson's Correlation:",cbar_title_fontsize=10,ticks_font_size=10).opts(title='%s-%s'%(e_x,e_y),default_tools=["pan"]))
fc_plot_R

# ## 7.2 Scatter across a few R-based FC matrices

fc_plot_scatter_R = pn.FlexBox()
for p in ['e01|e03_vs_e03|e03']:
    p1 = tuple((p.split('_vs_')[1]).split('|'))
    p2 = tuple((p.split('_vs_')[0]).split('|'))
    aux_fc_01 = sym_matrix_to_vec(fc['R','Basic',p1].values,discard_diagonal=True)
    aux_fc_02 = sym_matrix_to_vec(fc['R','Basic',p2].values,discard_diagonal=True)
    aux_df = pd.DataFrame([aux_fc_01,aux_fc_02],index=p.split('_vs_')).T
    plot = aux_df.hvplot.scatter(x=p.split('_vs_')[1],y=p.split('_vs_')[0], datashade=True,aspect='square') * hv.Slope(1,0).opts(line_color='k',line_dash='dashed',line_width=1)
    fc_plot_scatter_R.append(plot)

fc_plot_scatter_R

# # 8. C-based Results
#
# ## 8.1 FC matrices for all 9 echo pairs
#
# Only two of these are used in Figure 2.B

fc_plot_C = pn.FlexBox()
for (e_x,e_y) in echo_pairs_tuples:
    aux_fc = fc['C','Basic',(e_x,e_y)]
    fc_plot_C.append(hvplot_fc(aux_fc,
          major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', cbar_title="Covariance:",cbar_title_fontsize=10,ticks_font_size=10).opts(title='%s-%s'%(e_x,e_y),default_tools=["pan"]))
fc_plot_C

# ## 4.2 Scatter across a few C-FC matrices

echoes_dict = {'e01':13.7,'e02':30,'e03':47}
ideal_slopes = {}
for p in pairs_of_echo_pairs:
    x,y = p.split('_vs_')
    x_e1,x_e2 = x.split('|')
    y_e1,y_e2 = y.split('|')
    ideal_slopes[p] = (echoes_dict[y_e1] * echoes_dict[y_e2]) / (echoes_dict[x_e1] * echoes_dict[x_e2])
print(ideal_slopes)

fc_plot_scatter_C = pn.FlexBox()
for p in ['e01|e03_vs_e03|e03']:
    p1 = tuple((p.split('_vs_')[1]).split('|'))
    p2 = tuple((p.split('_vs_')[0]).split('|'))
    aux_fc_01 = sym_matrix_to_vec(fc['C','Basic',p1].values,discard_diagonal=True)
    aux_fc_02 = sym_matrix_to_vec(fc['C','Basic',p2].values,discard_diagonal=True)
    emp_slope,emp_int = np.polyfit(aux_fc_02,aux_fc_01,1)
    aux_df = pd.DataFrame([aux_fc_01,aux_fc_02],index=p.split('_vs_')).T
    plot = aux_df.hvplot.scatter(x=p.split('_vs_')[1],y=p.split('_vs_')[0], datashade=True,aspect='square', xlim=(-.1,.6), ylim=(-.5,3)) * hv.Slope(1,0).opts(line_color='r',line_dash='dashed',line_width=3) * \
            hv.Slope(ideal_slopes[p],0).opts(line_color='g',line_dash='dashed',line_width=3) * \
            hv.Slope(emp_slope,emp_int).opts(line_color='b',line_dash='dashed',line_width=3)
    fc_plot_scatter_C.append(plot)

fc_plot_scatter_C


