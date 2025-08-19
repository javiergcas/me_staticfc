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

# # Description: Spreng Dataset - FC Results
#
# This notebook allow us to investigate each scan of the gated/non-gated dataset and how C and R can be used to quantify data quality

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

#import os
#port_tunnel = int(os.environ['PORT2'])
#print('++ INFO: Second Port available: %d' % port_tunnel)
port_tunnel = 45719

# +
from utils.basics import TES_MSEC
from utils.basics import ATLASES_DIR, PRCS_DATA_DIR
import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import pickle
import panel as pn
from nilearn.connectome import sym_matrix_to_vec
from utils.basics import compute_residuals, echo_pairs, pairs_of_echo_pairs, echo_pairs_tuples, get_dataset_index
from utils.dashboard import fc_across_echoes_scatter_page, get_fc_matrices, get_static_report, get_fc_matrix,dynamic_summary_plot_gated

from utils.basics import mse_dist, chord_distance_between_intersecting_lines


# -

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


# ***
#
# # Select the dataset you want to work this next:

DATASET = input ('Select Dataset (discovery or evaluation):')

echo_times_dict = TES_MSEC[DATASET]
print(echo_times_dict)

ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())

# # 1. Basic Information
# Create lists with all 6 possible echo combinations, and then all possible pairings between those.

print('Echo Pairs[n=%d] = %s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d] = %s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

# ***
# # 2. Load Atlas Information

ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

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

# ***
# # 3. Load Timeseries and compute R and C matrices
#
# This cell will load ROI Timeseries, compute R and C, and place these into a dictionary of datafrmes. It will do this for the Basic denoising pipeline (Basic) and no censoring (ALL).

# +
pp_opts = {'No Censoring | Basic':'ALL_Basic',
           'No Censoring | GSR':'ALL_GS',
           'No Censoring | Tedana (fastica)':'ALL_Tedana-fastica', 
           'No Censoring | Tedana (fastica MDL)':'ALL_Tedana-fastica-mdl',
           'No Censoring | Tedana (robustica)':'ALL_Tedana-robustica'}
nordic_opts = {'Do not use':'off', 'Active':'on'}

data_fc = {}
# -

# %%time
filename = f'./cache/{DATASET}_fc.pkl'
i=0
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    with open(filename, 'rb') as f:
        data_fc = pickle.load(f)
else:
    for sbj,ses in tqdm(list(ds_index)):
        for nordic in nordic_opts.values():
            for pp in pp_opts.values():
                for (e_x,e_y) in echo_pairs_tuples:
                    d_folder = f'D03_Preproc_{ses}_NORDIC-{nordic}'
                    # Compose path to input TS
                    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{pp}.{ATLAS_NAME}_000.netts')
                    # Load TS into memory
                    if (not osp.exists(roi_ts_path_x)) or (not osp.exists(roi_ts_path_y)):
                        print(f'++ WARNING: Missing input files for {sbj},{ses},{e_x},{e_y},{nordic},{pp}')
                        print(f'            {roi_ts_path_x}')
                        print(f'            {roi_ts_path_y}')
                        i+=1
                        continue
                    roi_ts_x      = np.loadtxt(roi_ts_path_x)
                    roi_ts_y      = np.loadtxt(roi_ts_path_y)
                    aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
                    aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
                    # Compute the full correlation matrix between aux_ts_x and aux_ts_y
                    aux_r    = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
                    aux_c    = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
                    data_fc[sbj, ses, pp,nordic,'|'.join((e_x,e_y)),'R']  = pd.DataFrame(aux_r,index=roi_idxs,columns=roi_idxs)
                    data_fc[sbj, ses, pp,nordic,'|'.join((e_x,e_y)),'C']  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)
    with open(filename, 'wb') as f:
        pickle.dump(data_fc, f)
print(i)

# ***
#
# # 4. Compute pBOLD
#
# ## 4.1. First compute pBOLD on each separate scatter plot

# %%time
filename = f'./cache/{DATASET}_pBOLD_all_scatters.nc'
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    pBOLD_xr = xr.open_dataarray(filename)
else:
    pBOLD_xr = xr.DataArray(dims=['sbj','ses','pp','nordic','fc_metric','ee_vs_ee','qc_metric',],
                         coords={'sbj':       sbj_list,
                                 'ses':       ses_list,
                                 'pp':        list(pp_opts.values()),
                                 'nordic':    list(nordic_opts.values()),
                                 'fc_metric': ['R','C'],
                                 'ee_vs_ee':  pairs_of_echo_pairs,
                                 'qc_metric': ['pBOLD','pSo']})
    for sbj in tqdm(sbj_list):
        for ses in ses_list:
            partial_key = (sbj, ses)
            sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)
            if not sbj_ses_in_fc:
                print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))
                continue
            for fc_metric in ['C','R']:
                for pp in pp_opts.values():
                    for nordic in nordic_opts.values():
                        for eep in pairs_of_echo_pairs:
                            # Extract vectorized FC for this particular case
                            # ==============================================
                            eep1,eep2 = eep.split('_vs_')
                            data_df = pd.DataFrame(columns=[eep1,eep2])
                            data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)
                            data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)
                            
                            # Calculate slope and intercept for the two extreme scenarios
                            # ===========================================================
                            So_line_sl, So_line_int = 1.,0. # This is always the same
                            BOLD_line_int = 0.              # This is always the same
                            if fc_metric  == 'R':
                               BOLD_line_sl = 1.
                            if fc_metric == 'C':
                                e1_X,e2_X     = eep1.split('|')
                                e1_Y,e2_Y     = eep2.split('|')
                                BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])
                            # QC1. Compute dBOLD and dSo metrics
                            # ==================================
                            pBOLD_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'pBOLD'],pBOLD_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'pSo'] = mse_dist(data_df.values,
                                                                                                                                           BOLD_line_sl,
                                                                                                                                           So_line_sl, 
                                                                                                                                           weight_fn=lambda r: np.power(r,1.0), 
                                                                                                                                           max_weight_fn=lambda r: np.minimum(r,np.quantile(r,.95)),
                                                                                                                                           tol=1e-3)
    pBOLD_xr.to_netcdf(filename)

# ## 4.2. Weigthed average across scatter plots
#
# Becuase the separation of the BOLD and non-BOLD line is dependent on the contributing echoes, instead of simply averaging all scatter-specific pBOLD values, we propose to do a weigthed average where the weights correspond to the chord distance between the lines at radius = 1.0.
#
# The next cell computes this distance for all available scatter plots.

scat_plot_weights = xr.DataArray(np.zeros(len(pairs_of_echo_pairs)),dims='ee_vs_ee',coords={'ee_vs_ee':pairs_of_echo_pairs})
for ppe in pairs_of_echo_pairs:
    ep1,ep2 = ppe.split('_vs_')
    ex1,ex2 = ep1.split('|')
    ey1,ey2 = ep2.split('|')
    this_case_BOLD_slope = (echo_times_dict[ey1] * echo_times_dict[ey2]) / (echo_times_dict[ex1] * echo_times_dict[ex2])
    scat_plot_weights.loc[ppe] = chord_distance_between_intersecting_lines(1.0, this_case_BOLD_slope, r=1.0)

scat_plot_weights

# Here, we know calculate the final metrics per dataset. Only pBOLD will use the weights during the average. TSNR will be simply averaged.

QC_metrics = {}
for fc_metric in ['R','C']:
    for qc_metric in ['pBOLD','pSo']:
        aux_df = pBOLD_xr.weighted(scat_plot_weights).mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
        aux_df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
        QC_metrics[(fc_metric,qc_metric)] = aux_df
QC_metrics['C','pBOLD'].head(5)

# # 5. Gather TSNR information

# %%time
for TSNR_metric in ['TSNR (Full Brain)','TSNR (Visual Cortex)']:
    aux_df = pd.DataFrame(columns=['Subject','Session','Pre-processing','NORDIC',TSNR_metric])
    aux_df.set_index(['Subject','Session','Pre-processing','NORDIC'], inplace=True)
    for sbj in tqdm(sbj_list, desc=TSNR_metric):
        for ses in ses_list:
            partial_key = (sbj, ses)
            sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)
            if not sbj_ses_in_fc:
                print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))
                continue
            for pp in pp_opts.values():
                for nordic in nordic_opts.values():            
                    d_folder = f'D03_Preproc_{ses}_NORDIC-{nordic}'
                    if TSNR_metric == 'TSNR (Visual Cortex)':
                        aux_rois_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tsnr_stats_regress',f'TSNR_ROIs_e02_{pp}.txt')
                        aux_rois      = pd.read_csv(aux_rois_path,skiprows=3, sep=r'\s+').drop(0).set_index('ROI_name')
                        aux_df.loc[sbj,ses,pp,nordic] = float(aux_rois.loc['GHCP-R_Primary_Visual_Cortex','Tmed'])
                    if TSNR_metric == 'TSNR (Full Brain)':
                        aux_fb_path   = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tsnr_stats_regress',f'TSNR_FB_e02_{pp}.txt')
                        aux_fb        = pd.read_csv(aux_fb_path,skiprows=3, sep=r'\s+').drop(0).set_index('ROI_name')
                        aux_df.loc[sbj,ses,pp,nordic] = float(aux_fb.loc['NONE','Tmed'])
    QC_metrics['C',TSNR_metric] = aux_df.reset_index()
    QC_metrics['R',TSNR_metric] = aux_df.reset_index()

QC_metrics['C',qc_metric].head(5)

# ***
# # 6. Tedana derived metrics

# %%time
other_stats = pd.DataFrame(columns=['Subject','Session','NORDIC','Tedana Type','Component Type','Statistic','Value'])
other_stats = other_stats.set_index(['Subject','Session','NORDIC','Tedana Type','Component Type','Statistic'])
for sbj in tqdm(sbj_list, desc='Subjects'):
    for ses in ses_list:
        partial_key = (sbj, ses)
        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)
        if not sbj_ses_in_fc:
            print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))
            continue
        for nordic in nordic_opts.values():
            d_folder = f'D03_Preproc_{ses}_NORDIC-{nordic}'
            for tedana_type in ['fastica']:#,'robustica']:
                ica_metrics_path         = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'tedana_{tedana_type}','ica_metrics.tsv')
                ica_metrics              = pd.read_csv(ica_metrics_path, sep='\t').set_index('Component')
                likely_bold_components   = list(ica_metrics[ica_metrics['classification_tags']=='Likely BOLD'].index)
                unlikely_bold_components = list(ica_metrics[ica_metrics['classification_tags']=='Unlikely BOLD'].index)

                other_stats.loc[sbj,ses,nordic,tedana_type,'Likely BOLD','Summed Variance']   = ica_metrics.loc[likely_bold_components,'variance explained'].sum().round(2)
                other_stats.loc[sbj,ses,nordic,tedana_type,'Unlikely BOLD','Summed Variance'] = ica_metrics.loc[unlikely_bold_components,'variance explained'].sum().round(2)
                other_stats.loc[sbj,ses,nordic,tedana_type,'Likely BOLD','#ICs']              = len(likely_bold_components)
                other_stats.loc[sbj,ses,nordic,tedana_type,'Unlikely BOLD','#ICs']            = len(unlikely_bold_components)

# ***
#
# # Create Dashboard

# +
avial_qc_metrics = list(set([key[1] for key in QC_metrics.keys()]))

sbj_select                              = pn.widgets.Select(name='Subject',        options=sbj_list, width=200)
ses_select                              = pn.widgets.Select(name='Data Type',      options=ses_list+['all'], width=200)
pp_select                               = pn.widgets.Select(name='Pre-processing', options=pp_opts, width=200)
nordic_select                           = pn.widgets.Select(name='NORDIC',         options=nordic_opts, width=200)
fc_select                               = pn.widgets.Select(name='FC Metric',      options={'Correlation':'R','Covariance':'C'}, width=200)
plot_select                             = pn.widgets.Select(name='Plot type',      options={'Scatter Plot':'scatter','Hex Bin':'hexbin','FC Matrices across pipelines':'FCmats',
                                                                                            'FC Matrices across echoes':'FCmats_echoes',
                                                                                            'Group Results (Static)':'group_res_static', 'Group Results (Dynamic)':'group_res_dynamic'})

scat_lim_input                          = pn.widgets.FloatInput(name='Scatter Limit Value', value=1., step=0.1, start=0., end=50., width=200)
show_line_fit_checkbox                  = pn.widgets.Toggle(name='Show Linear Fit', button_type='primary')
scatter_extra_confs_card                = pn.Card(scat_lim_input,show_line_fit_checkbox, title='Scatter Plot & FCs | Configuration')

qc_metric_select                        = pn.widgets.Select(name='QC Metric to show', options=avial_qc_metrics, value='pBOLD', width=200)
pps_to_include_in_group_results         = pn.widgets.MultiSelect(name='Pipelines to include in Group Results', options=pp_opts, value=list(pp_opts.values())[0:3], width=200)
remove_outliers_from_swarm_plots_toggle = pn.widgets.Toggle(name='Remove Outliers from BarPlot', button_type='primary')
show_stats_toggle                       = pn.widgets.Toggle(name='Show Statistical Annotations', button_type='primary')
show_points_toggle                      = pn.widgets.Toggle(name='Show Individual Points', button_type='primary')
stat_test_select                        = pn.widgets.Select(name='Statistical Test', options={'Paired T-test':'t-test_paired','Independent T-test':'t-test_ind','Mann Whitney (Ind,non-param)':'Mann-Whitney'})
annot_type_select                       = pn.widgets.Select(name='Annotation Type', options={'Stars':'star','Simple Annotation':'simple','Full Annotation':'full'})
barplot_extra_confs_card                = pn.Card(qc_metric_select, show_points_toggle, show_stats_toggle,
                                                  stat_test_select, annot_type_select, pps_to_include_in_group_results,
                                                  remove_outliers_from_swarm_plots_toggle, title='Group Results (Static) | Configuration')

sidebar = [sbj_select,ses_select,pp_select,nordic_select,fc_select, pn.layout.Divider(),
           plot_select,pn.layout.Divider(),
           scatter_extra_confs_card,pn.layout.Divider(),
           barplot_extra_confs_card,
           ]


# -

@pn.depends(sbj_select,ses_select, pp_select, nordic_select, fc_select, plot_select, show_line_fit_checkbox, scat_lim_input,show_stats_toggle,stat_test_select,annot_type_select,pps_to_include_in_group_results,remove_outliers_from_swarm_plots_toggle, qc_metric_select,show_points_toggle)
def get_main_frame(sbj,ses, pp, nordic, fc_metric, plot_type, show_line_fit, ax_lim,show_stats,stat_test,annot_type,pps_to_include_in_barplot,remove_outliers_from_swarm_plots, qc_metric,show_points):
    if plot_type == 'hexbin':
        frame = fc_across_echoes_scatter_page(DATASET,data_fc,pBOLD_xr,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, other_stats=other_stats.loc[sbj,ses,nordic,:,:,:], hexbin=True)
        return frame
    if plot_type == 'scatter':
        frame = fc_across_echoes_scatter_page(DATASET,data_fc,pBOLD_xr,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, other_stats=other_stats.loc[sbj,ses,nordic,:,:,:], hexbin=False)
        return frame
    if plot_type == 'FCmats':
        fcR = get_fc_matrices(data_fc,pBOLD_xr,sbj,ses, nordic, 'R', net_cmap=power264_nw_cmap)
        fcC = get_fc_matrices(data_fc,pBOLD_xr,sbj,ses, nordic, 'C', net_cmap=power264_nw_cmap)
        return pn.Column(fcR,fcC)
    if plot_type == 'FCmats_echoes':
        layout = pn.GridBox(ncols=3)
        for ep in echo_pairs:
            fcR = get_fc_matrix(data_fc,pBOLD_xr,sbj,ses,pp,nordic,fc_metric,echo_pair=ep, net_cmap=power264_nw_cmap, ax_lim=ax_lim, title='%s | %s' % (fc_metric,ep))
            layout.append(fcR)
        return layout
    if plot_type == 'group_res_static':
        data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        aa = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  hue='Pre-processing',x='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left',show_points=show_points, session=ses),    title=f'{qc_metric} grouped by NORDIC')
        bb = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  x='Pre-processing',hue='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left', show_points=show_points, session=ses),   title=f'{qc_metric} grouped by Pre-processing')
        return pn.GridBox(*[aa,bb],ncols=2)
    if plot_type == 'group_res_dynamic':
        if fc_metric == 'C':
            pBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, fc_metric, 'pBOLD', nordic),title='pBOLD')
            return pn.Row(pBOLD_card,None,None)
        else:
            return pn.pane.Markdown('# This is not available for R-based FC')


template = pn.template.BootstrapTemplate(title=f'{DATASET} Dataset | Edge-based Results', 
                                         sidebar=sidebar,
                                         main=get_main_frame)

dashboard = template.show(port=port_tunnel, open=False)

# ***

dashboard.stop()

# ***

import holoviews as hv

sbj = 'MGSBJ03'
ses = 'cardiac_gated'
pp  = 'ALL_GS'
nordic='off'
eep1,eep2 = 'e01|e01','e01|e02'
fc_metric= 'C'

data_df = pd.DataFrame(columns=[eep1,eep2])
data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)
data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)

# +
So_line_sl, So_line_int = 1.,0. # This is always the same
BOLD_line_int = 0.              # This is always the same
if fc_metric  == 'R':
    BOLD_line_sl = 1.
if fc_metric == 'C':
    e1_X,e2_X     = eep1.split('|')
    e1_Y,e2_Y     = eep2.split('|')
    BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])

BOLD_line = hv.Slope(BOLD_line_sl,BOLD_line_int).opts(line_color='g',line_width=1)
So_line   = hv.Slope(So_line_sl,So_line_int).opts(line_color='r',line_width=1)
zero_x    = hv.HLine(0).opts(line_color='k',line_width=.5,line_dash='dashed')
zero_y    = hv.VLine(0).opts(line_color='k',line_width=.5,line_dash='dashed')
# -

mse_stats = mse_dist(data_df.values,BOLD_line_sl,So_line_sl, weight_fn=lambda r: np.power(r,.5), max_weight_fn=lambda r: np.minimum(r,np.quantile(r,.95)), verbose_return=True)
mse_stats

data_df['w']     = mse_stats['w']
data_df['dBOLD'] = mse_stats['d1']
data_df['dSo']   = mse_stats['d2']
print(data_df['w'].max())

(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='w', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['w'].min(),data_df['w'].max())) * BOLD_line * So_line * zero_x * zero_y) + \
(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='dBOLD', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['dBOLD'].quantile(0.05),data_df['dBOLD'].quantile(0.85))) * BOLD_line * So_line * zero_x * zero_y)

(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='w', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['w'].quantile(0.05),data_df['w'].quantile(0.85))) * BOLD_line * So_line * zero_x * zero_y) + \
(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='dBOLD', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['dBOLD'].quantile(0.05),data_df['dBOLD'].quantile(0.85))) * BOLD_line * So_line * zero_x * zero_y)


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


# +
def angdiff(a, b):
    """Smallest signed angle difference a-b in [-pi, pi]."""
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

def angdiff_line(theta, phi):
    """
    Smallest angle difference between a direction (theta)
    and an *undirected line* with orientation phi.
    Returns value in [0, pi/2].
    """
    # Wrap into [-pi, pi]
    d = (theta - phi + np.pi) % (2*np.pi) - np.pi
    # Fold over pi to remove direction
    return np.minimum(np.abs(d), np.pi - np.abs(d))

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
    return {'p_line1':frac_line1,
            'p_line2':frac_line2,
            'd1':d1,
            'd2':d2,
            'w':w}


# -

(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='w', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['w'].quantile(0.05),data_df['w'].quantile(0.85))) * BOLD_line * So_line * zero_x * zero_y) + \
(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='dBOLD', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['dBOLD'].quantile(0.01),data_df['dBOLD'].quantile(0.99))) * BOLD_line * So_line * zero_x * zero_y) + \
(data_df.hvplot.scatter(x=eep1,y=eep2, aspect='square',color='dSo', cmap='viridis',s=1, xlim=(-5,5),ylim=(-5,5)).opts(clim=(data_df['dSo'].quantile(0.01),data_df['dSo'].quantile(0.99))) * BOLD_line * So_line * zero_x * zero_y) 

data_df.hvplot.scatter(x='dBOLD',y='dSo', aspect='square',color='w', cmap='viridis',s=1)

data_df.hvplot.scatter(x='wdBOLD',y='wdSo', aspect='square',color='w', cmap='viridis',s=1)

points = data_df.values
points[:,0].shape



mse_dist(points,BOLD_line_sl,So_line_sl,weight_fn=lambda r: np.power(r,.5), max_weight=2.0)


# +
# Alternative metrics
def angdiff(a, b):
    """Smallest signed angle difference a-b in [-pi, pi]."""
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

def weighted_line_preference(points, m1, m2, weight_fn=None):
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
    w = np.minimum(2.,w)
    phi1 = np.arctan(m1)
    phi2 = np.arctan(m2)
    d1 = np.abs(angdiff(theta, phi1))
    d2 = np.abs(angdiff(theta, phi2))
    # prefer the line with smaller angular error
    pref1 = (d1 < d2).astype(float)
    # tie-breaker: if equal within tolerance, split 0.5
    tol = 1e-12
    ties = np.isclose(d1, d2, atol=tol)
    pref1[ties] = 0.5
    weighted_pref1 = (w * pref1).sum()
    total_weight = w.sum()
    frac_line1 = weighted_pref1 / (total_weight + 1e-16)
    return {
        'd1':d1,
        'd2':d2,
        'frac_line1': frac_line1,
        'weighted_pref1': weighted_pref1,
        'total_weight': total_weight,
        'per_point_weights': w,
        'per_point_pref1': pref1
    }


# -

np.minimum(2.,out['per_point_weights'])

















# ***
#
# # Figures for OHBM poster

import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from itertools import combinations
def get_barplot(qa_xr,nordic,fc_metric,qc_metric,x='Pre-processing',hue='NORDIC',show_stats=False, stat_test='t-test_paired',stat_annot_type='star', legend_location='best', remove_outliers_from_swarm=True):
    """
    Create Static Bar Graph for a given quality metric
    """
    df         = qa_xr.mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
    df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
    df         = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana','ALL_Tedana-NORDIC_FixNComps':'Tedana (n=88)', 'NORDIC':'On'})
    num_hues   = len(list(df[hue].unique()))
    df_swarm = df.copy()
    if remove_outliers_from_swarm:
        quantile_value = df[qc_metric].quantile(.97)
        df_swarm[qc_metric]=df_swarm[qc_metric].where(df_swarm[qc_metric] <= quantile_value, np.nan)

    if (x=='Pre-processing') and (hue=='NORDIC'):
        pairs  = [((p,'On'),(p,'Off')) for p in df[x].unique()]
        colors = sns.color_palette("rocket",num_hues)
    if (x=='NORDIC') and (hue=='Pre-processing'):
        pairs      = [(('On',c[0]),('On',c[1])) for c in combinations(list(df['Pre-processing'].unique()),2)]
        pairs      = pairs + [(('Off',c[0]),('Off',c[1])) for c in combinations(list(df['Pre-processing'].unique()),2)]
        colors = sns.color_palette("Set2",num_hues)

    sns.set_context("paper", rc={"xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 16, 'legend.fontsize':16})
    fig, axs = plt.subplots(1,1,figsize=(6,6));
    sns.despine(top=True, right=True)
    sns.barplot(data=df,hue=hue, y=qc_metric, x=x, alpha=0.5, ax =axs, errorbar=('ci',95), palette=colors);
    sns.swarmplot(data=df_swarm,hue=hue, y=qc_metric, x=x, ax =axs, s=.5, dodge=True, legend=False, palette=colors);
    
    if show_stats:
        annotation = Annotator(axs, pairs, data=df, x=x, y=qc_metric, hue=hue);
        annotation.configure(test=stat_test, text_format=stat_annot_type, loc='inside', verbose=0);
        annotation.apply_test(alternative='two-sided');
        annotation.annotate();
    sns.move_legend(axs, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,)
    plt.tight_layout()
    plt.close()
    return fig


a = get_barplot(qa_xr.sel(pp=['ALL_Basic','ALL_GSasis','ALL_Tedana']),'Off','C','pBOLD',hue='Pre-processing',x='NORDIC',stat_test='t-test_paired', show_stats=True, stat_annot_type='star', legend_location='lower left', )

a

a.savefig('./saved_images/pBOLD_evaluation_group_result.eps')

a = get_barplot(qa_xr.sel(pp=['ALL_Basic','ALL_GSasis','ALL_Tedana']),'Off','C','TSNR (Full Brain)',hue='Pre-processing',x='NORDIC',stat_test='t-test_paired', show_stats=True, stat_annot_type='star', legend_location='lower left', )
a

a.savefig('./saved_images/TSNR_evaluation_group_result.eps')

a = get_barplot(qa_xr.sel(pp=['ALL_Basic','ALL_GSasis','ALL_Tedana','ALL_Tedana-NORDIC_FixNComps']),'On','C','TSNR (Full Brain)',x='Pre-processing',hue='NORDIC',stat_test='t-test_paired', show_stats=True, stat_annot_type='star', legend_location='lower left', )


a

fc_metric, qc_metric = 'C','TSNR (Full Brain)'
df         = qa_xr.mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
df         = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana','ALL_Tedana-NORDIC_FixNComps':'Tedana (n=88)', 'NORDIC':'On'})

df['TSNR (Full Brain)'].replace() df['TSNR (Full Brain)'].quantile(.97)

quantile_value = df['TSNR (Full Brain)'].quantile(.97)
df['TSNR (Full Brain)'] = df['TSNR (Full Brain)'].where(df['TSNR (Full Brain)'] <= quantile_value, quantile_value)



