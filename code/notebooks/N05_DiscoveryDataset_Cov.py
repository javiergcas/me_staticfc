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
#     display_name: Generic Kernel (2025a)
#     language: python
#     name: generic_2025a
# ---

# # Description: Gated Data - Variance
#
# This notebook allow us to investigate each scan of the gated/non-gated dataset and how C and R can be used to quantify data quality

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# +
from utils.basics import TES_MSEC, SESSIONS
from utils.basics import ATLASES_DIR, PRCS_DATA_DIR, PRJ_DIR
import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from utils.basics import compute_residuals, softmax
from utils.dashboard import get_barplot_discovery_dataset, get_cov_heatmap, cov_across_echoes_scatter_page,dynamic_summary_plot_gated
import pickle

import holoviews as hv


# -

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


# ***

echo_times_dict = TES_MSEC['Gating']
ses_list        = SESSIONS['Gating']

ATLAS_NAME = 'Power264_GatingDataset'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

sbj_list = ['MGSBJ01',  'MGSBJ02',  'MGSBJ03',  'MGSBJ04',  'MGSBJ05',  'MGSBJ06',  'MGSBJ07']
ses_list = ['constant_gated', 'cardiac_gated']
dataset_info_df = pd.DataFrame(index=pd.MultiIndex.from_product([sbj_list,ses_list],names=['Subject','Session']))

# # 1. Basic Information
# Create lists with all possible echo combinations, and then all possible pairings between those.

# +
from itertools import combinations_with_replacement, combinations

echo_pairs_tuples   = [i for i in combinations(['e01','e02','e03'],2)]
echo_pairs          = [('|').join(i) for i in echo_pairs_tuples]
pairs_of_echo_pairs = ['|'.join((e_x[0],e_x[1]))+'_vs_'+'|'.join((e_y[0],e_y[1])) for e_x,e_y in combinations(echo_pairs_tuples,2)]
# -

print('Echo Pairs[n=%d] = %s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d] = %s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

# ***
# # 2. Load Atlas Information

# +
roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df.head(5)

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)

print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index
# -

roi_info_df

# Create a dictionary to be used as colormap when plotting FC matrices

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

# ***
# # 3. Load Timeseries and compute Covariance
#
# This cell will load ROI Timeseries, compute covariance of all ROIs and place these into a dictionary of datafrmes. It will do this for the Basic denoising pipeline (Basic) and no censoring (ALL).

pp_opts = {'No Censoring | Basic':'ALL_Basic',
       'No Censoring | GSR':'ALL_GSasis',
       'No Censoring | Tedana':'ALL_Tedana'}
nordic_opts={'Do not use':'Off', 'Active':'On'}
data_cvar = {}

for sbj in tqdm(sbj_list, desc='Subject'):
    for ses in ses_list:
        for pp in pp_opts.values():
            for nordic in nordic_opts.values():
                # Compose Dfolder name
                if nordic == 'Off':
                    d_folder = f'D02_Preproc_fMRI_{ses}'
                if nordic == 'On':
                    d_folder = f'D04_Preproc_fMRI_{ses}_NORDIC'
                #if nordic == 'NORDIC_FixNComps':
                #    d_folder = f'D05_Preproc_fMRI_{ses}_{nordic}'
                for (e_x,e_y) in echo_pairs_tuples:
                    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    roi_ts_x      = np.loadtxt(roi_ts_path_x)
                    roi_ts_y      = np.loadtxt(roi_ts_path_y)
                    Nt,_ = roi_ts_x.shape
                    data_cvar[sbj, ses, pp,nordic,'|'.join((e_x,e_y)),'C']  = (roi_ts_x * roi_ts_y).sum(axis=0)/(Nt-1)

roi_ts_x.shape, roi_ts_y.shape,((roi_ts_x * roi_ts_y).sum(axis=0)/(Nt-1)).shape

# ***
#
# # 4. Compute QA-metrics

qa_xr = xr.DataArray(dims=['sbj','ses','pp','nordic','fc_metric','ee_vs_ee','qc_metric',],
                     coords={'sbj':sbj_list,
                             'ses':ses_list,
                             'pp': list(pp_opts.values()),
                             'nordic':list(nordic_opts.values()),
                             'fc_metric':['C'],
                             'ee_vs_ee':pairs_of_echo_pairs,
                             'qc_metric':['dBOLD','dSo','pBOLD','pSo']})

for sbj in tqdm(sbj_list):
    for ses in ses_list:
        partial_key = (sbj, ses)
        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_cvar)
        if not sbj_ses_in_fc:
            print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))
            continue
        for pp in pp_opts.values():
            for nordic in nordic_opts.values():
                for eep in pairs_of_echo_pairs:
                    # Extract vectorized FC for this particular case
                    eep1,eep2 = eep.split('_vs_')
                    data_df = pd.DataFrame(columns=[eep1,eep2])
                    data_df[eep1] = data_cvar[sbj,ses,pp,nordic,eep1,'C']
                    data_df[eep2] = data_cvar[sbj,ses,pp,nordic,eep2,'C']

                    # Calculate slope and intercept for the two extreme scenarios
                    So_line_sl, So_line_int = 1.,0. # This is always the same
                    BOLD_line_int = 0.              # This is always the same

                    e1_X,e2_X     = eep1.split('|')
                    e1_Y,e2_Y     = eep2.split('|')
                    BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])

                    # Compute dBOLD and dSo metrics
                    # WEIGTHED BY DIST TO ORIGIN: qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dBOLD'] = np.sqrt(((np.sqrt(np.power(data_df,2).sum(axis=1)).values * compute_residuals(data_df[eep1].values,data_df[eep2].values,BOLD_line_sl,BOLD_line_int))**2).sum())
                    # WEIGTHED BY DIST TO ORIGIN:qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dSo']   = np.sqrt(((np.sqrt(np.power(data_df,2).sum(axis=1)).values * compute_residuals(data_df[eep1].values,data_df[eep2].values,So_line_sl,So_line_int))**2).sum())
                    # REMOVE OUTLIERS: qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dBOLD'] = np.sqrt((reject_outliers(compute_residuals(data_df[eep1].values,data_df[eep2].values,BOLD_line_sl,BOLD_line_int),3)**2).sum())
                    # REMOVE OUTLIERS: qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dSo']   = np.sqrt((reject_outliers(compute_residuals(data_df[eep1].values,data_df[eep2].values,So_line_sl,  So_line_int),3)**2).sum())
                    qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dBOLD'] = np.sqrt((compute_residuals(data_df[eep1].values,data_df[eep2].values,BOLD_line_sl,BOLD_line_int)**2).sum())
                    qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dSo']   = np.sqrt((compute_residuals(data_df[eep1].values,data_df[eep2].values,So_line_sl,  So_line_int)**2).sum())
                    
                    # Compute probabilities
                    qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'pBOLD'], qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'pSo'] = 1 - softmax(qa_xr.loc[sbj,ses,pp,nordic,'C',eep,['dBOLD','dSo']].values)

# ***
# # Tedana derived metrics

# +
other_stats = {'Off': pd.DataFrame(columns=['Likely BOLD | Var','Unlikely BOLD | Var', 'Likely BOLD | #ICs','Unlikely BOLD | #ICs'], index=dataset_info_df.index),
               'On': pd.DataFrame(columns=['Likely BOLD | Var','Unlikely BOLD | Var', 'Likely BOLD | #ICs','Unlikely BOLD | #ICs'], index=dataset_info_df.index)}
# Results when NORDIC is Off
for sbj in sbj_list:
    for ses in ses_list:
        for nordic in nordic_opts.values():
            # Compose path to input TS < ==== This part will need to change once I get the tedana options in the correct place
            if nordic == 'Off':
                d_folder = f'D02_Preproc_fMRI_{ses}'
            if nordic == 'On':
                d_folder = f'D04_Preproc_fMRI_{ses}_NORDIC'
            ica_metrics_path         = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tedana_r01','ica_metrics.tsv')
            if not osp.exists(ica_metrics_path):
                print("++ WARNING: Missing data [%s,%s]" % (sbj,ses))
                continue
            ica_metrics              = pd.read_csv(ica_metrics_path, sep='\t').set_index('Component')
            likely_bold_components   = list(ica_metrics[ica_metrics['classification_tags']=='Likely BOLD'].index)
            unlikely_bold_components = list(ica_metrics[ica_metrics['classification_tags']=='Unlikely BOLD'].index)
            other_stats[nordic].loc[(sbj,ses),'Likely BOLD | Var']    = ica_metrics.loc[likely_bold_components,'variance explained'].sum().round(2)
            other_stats[nordic].loc[(sbj,ses),'Unlikely BOLD | Var']  = ica_metrics.loc[unlikely_bold_components,'variance explained'].sum().round(2)
            other_stats[nordic].loc[(sbj,ses),'Likely BOLD | #ICs']   = len(likely_bold_components)
            other_stats[nordic].loc[(sbj,ses),'Unlikely BOLD | #ICs'] = len(unlikely_bold_components)

other_stats['Off'] = other_stats['Off'].infer_objects()
other_stats['On']  = other_stats['On'].infer_objects()
# -

# ***
#
# # Create Dashboard

# +
sbj_select    = pn.widgets.Select(name='Subject',        options=sbj_list, width=200)
ses_select    = pn.widgets.Select(name='Data Type',      options=ses_list, width=200)
pp_select     = pn.widgets.Select(name='Pre-processing', options=pp_opts, width=200)
nordic_select = pn.widgets.Select(name='NORDIC',         options=nordic_opts, width=200)
fc_select     = pn.widgets.Select(name='FC Metric',      options={'Correlation':'R','Covariance':'C'}, width=200)
plot_select   = pn.widgets.Select(name='Plot type',      options={'Scatter Plot':'scatter','Carpet Plots of Regional Covariance':'carpet_plots',
                                                                 'Group Results (Static)':'group_res_static', 'Group Results (Dynamic)':'group_res_dynamic'})

scat_lim_input = pn.widgets.FloatInput(name='Scatter Limit Value', value=1., step=0.1, start=0., end=50., width=200)
show_line_fit_checkbox = pn.widgets.Toggle(name='Show Linear Fit', button_type='primary')
scatter_extra_confs_card = pn.Card(scat_lim_input,show_line_fit_checkbox, title='Scatter Plot & FCs | Configuration')

show_stats_toggle = pn.widgets.Toggle(name='Show Statistical Annotations', button_type='primary')
stat_test_select  = pn.widgets.Select(name='Statistical Test', options={'Paired T-test':'t-test_paired','Independent T-test':'t-test_ind','Mann Whitney (Ind,non-param)':'Mann-Whitney'})
annot_type_select = pn.widgets.Select(name='Annotation Type', options={'Stars':'star','Simple Annotation':'simple','Full Annotation':'full'})
barplot_extra_confs_card = pn.Card(show_stats_toggle,stat_test_select,annot_type_select, title='Group Results (Static) | Configuration')
sidebar = [sbj_select,ses_select,pp_select,nordic_select,fc_select, pn.layout.Divider(),
           plot_select,pn.layout.Divider(),
           scatter_extra_confs_card,pn.layout.Divider(),
           barplot_extra_confs_card,
           ]


# -

@pn.depends(sbj_select,ses_select, pp_select, nordic_select, fc_select, plot_select, show_line_fit_checkbox, scat_lim_input,show_stats_toggle,stat_test_select,annot_type_select)
def get_main_frame(sbj,ses, pp, nordic, fc_metric, plot_type, show_line_fit, ax_lim,show_stats,stat_test,annot_type):
    if plot_type == 'scatter':
        frame = cov_across_echoes_scatter_page(data_cvar,qa_xr,sbj,ses,pp, nordic, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, roi_info=roi_info_df, cmap=power264_nw_cmap, other_stats=other_stats[nordic])
        return frame
    if plot_type == 'carpet_plots':
        return get_cov_heatmap(data_cvar,sbj,ses,pp_opts,nordic_opts,roi_info=roi_info_df,clim=ax_lim, echo_pairs=echo_pairs)
    if plot_type == 'group_res_static':
        a = pn.Card(get_barplot_discovery_dataset(qa_xr,'On',fc_metric,'pBOLD',   hue='Pre-processing',x='Session',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, legend_location='lower left'),
                    get_barplot_discovery_dataset(qa_xr,'On',fc_metric,'pBOLD',  hue='Session',x='Pre-processing',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, legend_location='lower left'),
                    title='Results of Speng Sample (1) | NORDIC On')
        b = pn.Card(get_barplot_discovery_dataset(qa_xr,'Off',fc_metric,'pBOLD',   hue='Pre-processing',x='Session',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, legend_location='lower left'),
                    get_barplot_discovery_dataset(qa_xr,'Off',fc_metric,'pBOLD',  hue='Session',x='Pre-processing',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, legend_location='lower left'),
                    title='Results of Speng Sample (1) | NORDIC Off')
        return pn.Row(a,b)
        #return pn.Row(get_barplot(qa_xr,nordic,fc_metric,'pBOLD',  hue='Pre-processing',x='NORDIC',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type),
        #              get_barplot(qa_xr,nordic,fc_metric, 'dBOLD', hue='Pre-processing',x='NORDIC',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type),
        #              get_barplot(qa_xr,nordic,fc_metric, 'dSo',   hue='Pre-processing',x='NORDIC',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type))
    if plot_type == 'group_res_dynamic':
        if fc_metric == 'C':
            pBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, 'C', 'pBOLD', nordic),title='pBOLD')
            #dBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, 'C', 'dBOLD', nordic),title='dBOLD')
            #dS0_card   = pn.Card(dynamic_summary_plot_gated(qa_xr, 'C', 'dSo', nordic),title='dSo')
            return pn.Row(pBOLD_card,None,None)
            #return pn.Row(pBOLD_card,dBOLD_card,dS0_card)
        else:
            return pn.pane.Markdown('# This is not available for R-based FC')


template = pn.template.BootstrapTemplate(title='Discovery Dataset | Region-based Results', 
                                         sidebar=sidebar,
                                         main=get_main_frame)

dashboard = template.show(port=port_tunnel)

# ***

dashboard.stop()

# ***

from scipy.stats import pearsonr, spearmanr

sample_basic  = sym_matrix_to_vec(data_fc[('MGSBJ05', 'constant_gated', 'ALL_Basic', 'e01|e02', 'R')].values)
sample_gsr    = sym_matrix_to_vec(data_fc[('MGSBJ05', 'constant_gated', 'ALL_GSasis', 'e01|e02', 'R')].values)
sample_tedana = sym_matrix_to_vec(data_fc[('MGSBJ05', 'constant_gated', 'ALL_Tedana', 'e01|e02', 'R')].values)

fig,axs=plt.subplots(1,3, figsize=(20,5))
sns.heatmap(data_fc[('MGSBJ05', 'constant_gated', 'ALL_Basic', 'e01|e02', 'R')].values, cmap='RdBu_r', vmin=-.8, vmax=.8,ax=axs[0])
sns.heatmap(data_fc[('MGSBJ05', 'constant_gated', 'ALL_GSasis', 'e01|e02', 'R')].values, cmap='RdBu_r', vmin=-.8, vmax=.8,ax=axs[1])
sns.heatmap(data_fc[('MGSBJ05', 'constant_gated', 'ALL_Tedana', 'e01|e02', 'R')].values, cmap='RdBu_r', vmin=-.8, vmax=.8,ax=axs[2])

fig, axs = plt.subplots(1,3, figsize=(20,5))
sns.scatterplot(x=sample_basic,y=sample_gsr,s=1, ax=axs[0])
axs[0].set_xlim(-.8,.8)
axs[0].set_ylim(-.8,.8)
axs[0].set_box_aspect(1)
axs[0].axline([0, 0], [1, 1], linestyle='--', color='k', linewidth=.5)
sns.scatterplot(x=sample_basic,y=sample_tedana,s=1, ax=axs[1])
axs[1].set_xlim(-.8,.8)
axs[1].set_ylim(-.8,.8)
axs[1].set_box_aspect(1)
axs[1].axline([0, 0], [1, 1], linestyle='--', color='k', linewidth=.5)


pearsonr(sample_basic,sample_gsr), spearmanr(sample_basic,sample_gsr)

pearsonr(sample_basic,sample_tedana)

pearsonr(sample_gsr,sample_tedana)

# ***

# # Group Level Results

df= qa_xr.sel(fc_metric='C', qc_metric='pBOLD').mean(dim='ee_vs_ee').to_dataframe(name='pBOLD').drop(['fc_metric','qc_metric'],axis=1).reset_index()
df.columns=['Subject','Data Type','Pre-processing','pBOLD']
df = df.replace({'constant_gated':'Constant TR','cardiac_gated':'Cardiac Gating', 'ALL_Basic':'Basic Regressors','ALL_GSasis':'GSR','ALL_Tedana':'Tedana'})
df.head(3)

g = sns.catplot(data=df,kind='bar',x='Data Type',hue='Pre-processing',y='pBOLD', errorbar=('ci', 95), alpha=0.5)
sns.swarmplot(data=df, x="Data Type",hue='Pre-processing', y="pBOLD", size=3, dodge=True, legend=False)
g.set_axis_labels("", "Prob (BOLD dominated)")
g.despine(left=True)

df= qa_xr.sel(fc_metric='C', qc_metric='pBOLD',pp=['ALL_Tedana','ALL_Tedana_NORDIC']).mean(dim='ee_vs_ee').to_dataframe(name='pBOLD').drop(['fc_metric','qc_metric'],axis=1).reset_index()
df.columns=['Subject','Data Type','Apply NORDIC?','pBOLD']
df = df.replace({'constant_gated':'Constant TR','cardiac_gated':'Cardiac Gating', 'ALL_Tedana':'NORDIC Off','ALL_Tedana_NORDIC':'NORDIC On'})
df.head(3)

g = sns.catplot(data=df,kind='bar',x='Data Type',hue='Apply NORDIC?',y='pBOLD', errorbar=('ci', 95), alpha=0.5, legend_out=False)
sns.swarmplot(data=df, x="Data Type",hue='Apply NORDIC?', y="pBOLD", size=3, dodge=True, legend=False)
g.set_axis_labels("", "Prob (BOLD dominated)")
g.despine()

df

data = {}
for sbj in tqdm(sbj_list, desc='Subject'):
    for ses in ses_list:
        for scenario in scenarios:
            aux = scenario.split('-')
            pp,nordic_conf=aux if len(aux)==2 else [aux[0],'Off']
            # Compose Dfolder name
            if nordic_conf == 'Off':
                d_folder = f'D02_Preproc_fMRI_{ses}'
            if nordic_conf == 'NORDIC':
                d_folder = f'D04_Preproc_fMRI_{ses}_{nordic_conf}'
            if nordic_conf == 'NORDIC_FixNComps':
                d_folder = f'D05_Preproc_fMRI_{ses}_{nordic_conf}'
            for e_x in ['e01','e02','e03']:
                roi_ts_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{pp}.{ATLAS_NAME}_000.netts')
                roi_ts      = np.loadtxt(roi_ts_path)
                Nt,_        = roi_ts.shape
                data[sbj, ses, pp,nordic_conf,e_x,'VAR']  = np.var(roi_ts,axis=0)

sbj = 'MGSBJ01'
ses = 'constant_gated'
pp = 'ALL_Tedana'
nordic = 'NORDIC'
df = pd.DataFrame([data[sbj,ses,pp,nordic,'e01','VAR'],data[sbj,ses,pp,nordic,'e02','VAR'],data[sbj,ses,pp,nordic,'e03','VAR']], 
             index=[echo_times_dict[e] for e in ['e01','e02','e03']]).T
df.columns.name = 'TE'
df = df.melt()
df.index = pd.concat([pd.DataFrame(index=roi_idxs),pd.DataFrame(index=roi_idxs),pd.DataFrame(index=roi_idxs)]).index
df['NW_Color'] = [power264_nw_cmap[n] for n in df.index.get_level_values('Network')]
df.hvplot.scatter(x='TE',y='value',hover_cols=['ROI_Name','Network','Hemisphere'], color='NW_Color')

sbj = 'MGSBJ01'
ses = 'cardiac_gated'
pp = 'ALL_Basic'
nordic = 'Off'
df = pd.DataFrame([data[sbj,ses,pp,nordic,'e01','VAR'],data[sbj,ses,pp,nordic,'e02','VAR'],data[sbj,ses,pp,nordic,'e03','VAR']], 
             index=[echo_times_dict[e] for e in ['e01','e02','e03']]).T
df.columns.name = 'TE'
df = df.melt()
df.index = pd.concat([pd.DataFrame(index=roi_idxs),pd.DataFrame(index=roi_idxs),pd.DataFrame(index=roi_idxs)]).index
df['NW_Color'] = [power264_nw_cmap[n] for n in df.index.get_level_values('Network')]
df.hvplot.scatter(x='TE',y='value',hover_cols=['ROI_Name','Network','Hemisphere'], color='NW_Color')

pd.DataFrame(data[('MGSBJ06', 'cardiac_gated', 'ALL_Tedana', 'Off', 'e01|e01', 'C')]).hvplot(label='e1') * \
pd.DataFrame(data[('MGSBJ06', 'cardiac_gated', 'ALL_Tedana', 'Off', 'e02|e02', 'C')]).hvplot(label='e2') * \
pd.DataFrame(data[('MGSBJ06', 'cardiac_gated', 'ALL_Tedana', 'Off', 'e03|e03', 'C')]).hvplot(label='e3')

pd.DataFrame(data[('MGSBJ06', 'cardiac_gated', 'ALL_Tedana', 'Off', 'e01|e02', 'C')]).hvplot(label='e01|e02') * \
pd.DataFrame(data[('MGSBJ06', 'cardiac_gated', 'ALL_Tedana', 'Off', 'e02|e03', 'C')]).hvplot(label='e02|e03') * \
pd.DataFrame(data[('MGSBJ06', 'cardiac_gated', 'ALL_Tedana', 'Off', 'e01|e03', 'C')]).hvplot(label='e01|e03')

path_e1 = '/data/SFIMJGC_HCP7T/BCBL2024/prcs_data/MGSBJ06/D02_Preproc_fMRI_cardiac_gated/errts.MGSBJ06.r01.e01.volreg.scale.tproject_ALL_Tedana.Power264_GatingDataset_000.netts'
e1_ts = pd.DataFrame(np.loadtxt(path_e1),columns=['ROI'+str(i).zfill(3) for i in range(226)])
path_e2 = '/data/SFIMJGC_HCP7T/BCBL2024/prcs_data/MGSBJ06/D02_Preproc_fMRI_cardiac_gated/errts.MGSBJ06.r01.e02.volreg.scale.tproject_ALL_Tedana.Power264_GatingDataset_000.netts'
e2_ts = pd.DataFrame(np.loadtxt(path_e2),columns=['ROI'+str(i).zfill(3) for i in range(226)])
path_e3 = '/data/SFIMJGC_HCP7T/BCBL2024/prcs_data/MGSBJ06/D02_Preproc_fMRI_cardiac_gated/errts.MGSBJ06.r01.e03.volreg.scale.tproject_ALL_Tedana.Power264_GatingDataset_000.netts'
e3_ts = pd.DataFrame(np.loadtxt(path_e3),columns=['ROI'+str(i).zfill(3) for i in range(226)])

e1_ts['ROI136'].hvplot(width=1000, label='e1') * e2_ts['ROI136'].hvplot(width=1000, label='e2') * e3_ts['ROI136'].hvplot(width=1000, label='e3')

e1_ts['ROI131'].hvplot(width=1000, label='e1') * e2_ts['ROI131'].hvplot(width=1000, label='e2') * e3_ts['ROI131'].hvplot(width=1000, label='e3')


