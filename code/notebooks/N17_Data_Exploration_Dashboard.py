#!/usr/bin/env python
# coding: utf-8

# # Description: Spreng Dataset - FC Results
# 
# This notebook allow us to investigate each scan of the gated/non-gated dataset and how C and R can be used to quantify data quality

# In[1]:


import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import pickle
import panel as pn
pn.extension
from nilearn.connectome import sym_matrix_to_vec


# In[2]:


from utils.basics import compute_residuals, echo_pairs, pairs_of_echo_pairs, echo_pairs_tuples, get_dataset_index, get_altas_info
from utils.basics import TES_MSEC
from utils.basics import ATLASES_DIR, CODE_DIR,PRCS_DATA_DIR, PRJ_DIR, FMRI_FINAL_NUM_SAMPLES, FMRI_TRS, NUM_DISCARDED_VOLUMES, DOWNLOAD_DIRS
from utils.basics import mse_dist, chord_distance_between_intersecting_lines, read_group_physio_reports
from utils.dashboard import fc_across_echoes_scatter_page, get_fc_matrices, get_static_report, get_fc_matrix,dynamic_summary_plot_gated, get_ts_report_page

from sklearn.ensemble import IsolationForest


# ***
# 
# # Select the dataset you want to work with:

# In[3]:


DATASET = input ('Select Dataset (discovery or evaluation):')


# In[4]:


DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]


# In[5]:


echo_times_dict = TES_MSEC[DATASET]
print(echo_times_dict)


# In[6]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# # 1. Basic Information
# Create lists with all 6 possible echo combinations, and then all possible pairings between those.

# In[7]:


print('Echo Pairs[n=%d] = %s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d] = %s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))


# ***
# # 2. Load Atlas Information

# In[8]:


ATLAS_NAME = f'Power264-{DATASET}'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)


# In[9]:


roi_info_df, power264_nw_cmap = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# In[10]:


pp_opts = {'No Censoring | No Regression':f'ALL_NoRegression',
           'No Censoring | Basic':f'ALL_Basic',
           'No Censoring | GSR':f'ALL_GS',
           'No Censoring | Tedana (fastica)':f'ALL_Tedana-fastica'}
nordic_opts = {'Do not use':'off', 'Active':'on'}
data_fc = {}


# ***
# # 3. Load FC matrices

# In[11]:


get_ipython().run_cell_magic('time', '', "fc_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_FC.pkl')\nprint('++ FC will be saved in %s' % fc_path)\n\nwith open(fc_path, 'rb') as f:\n    data_fc = pickle.load(f)\n")


# ***
# # 4. Load pBOLD

# In[12]:


pBOLD_xr_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_pBOLD.nc')
print('++ pBOLD will be loaded from %s' % pBOLD_xr_path)
pBOLD_xr = xr.open_dataarray(pBOLD_xr_path)


# In[13]:


QC_metrics = {}
for FC_METRIC in ['corr','cov']:
    pBOLD = pBOLD_xr.sel(fc_metric=FC_METRIC,ee_vs_ee='scan')
    pBOLD = pBOLD.to_dataframe(name='pBOLD').reset_index().drop(['qc_metric','fc_metric','ee_vs_ee'],axis=1)
    pBOLD.columns=['Subject','Session','Pre-processing','m-NORDIC','pBOLD']
    QC_metrics[(FC_METRIC,'pBOLD')] = pBOLD


# ***
# # 5. Load TSNR

# In[14]:


TSNR_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_TSNR.pkl')
print('++ TSNR will be loaded from %s' % TSNR_path)
with open(TSNR_path, 'rb') as f:
    TSNR = pickle.load(f)


# In[15]:


for key,val in TSNR.items():
    QC_metrics[key] = val


# ***
# # 6. Tedana derived metrics

# In[16]:


Tedana_QC_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_Tedana_QC.pkl')
print('++ Tedana statistics will be loaded from %s' % Tedana_QC_path)
with open(Tedana_QC_path, 'rb') as f:
    Tedana_QC = pickle.load(f)


# In[17]:


for key,val in Tedana_QC.items():
    new_key = (key[0].replace('C','cov'),key[1]) # I now use cov to refer to Covariance-based results instead of C
    QC_metrics[new_key] = val


# In[18]:


for tedana_metric in ['#ICs (All)','#ICs (Likely BOLD)','#ICs (Unlikely BOLD)','Var. Exp. (Likely BOLD)','Var. Exp. (Unlikely BOLD)']:
    QC_metrics['cov',tedana_metric] = QC_metrics['cov',tedana_metric].reset_index()
    QC_metrics['corr',tedana_metric] = QC_metrics['cov',tedana_metric].copy() # These metrics do not depend on FC choice. We duplicate entries so that selection is simpler later when constructing the dashboard


# ***
# 
# # 7. Physiological Recording Derived Metrics

# In[19]:


if DATASET == 'evaluation':
    report_card_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_card_review_all_scans.txt')
    report_card_summary_df    = read_group_physio_reports(report_card_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_card_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_card = report_card_summary_df['peak ival over dset mean std'].copy()
    df_card.columns=['Mean','St.Dev.']
    df_card['color'] = ['red' if c else 'green' for c in outliers]
    
    QC_metrics[('corr','Physio (cardiac)')] = df_card
    QC_metrics[('cov','Physio (cardiac)')] = df_card
else:
    QC_metrics[('corr','Physio (cardiac)')] = None
    QC_metrics[('cov','Physio (cardiac)')] = None


# In[20]:


if DATASET == 'evaluation':
    report_resp_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_resp_review_all_scans.txt')
    report_resp_summary_df    = read_group_physio_reports(report_resp_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_resp_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_resp = report_resp_summary_df['peak ival over dset mean std'].copy()
    df_resp.columns=['Mean','St.Dev.']
    df_resp['color'] = ['red' if c else 'green' for c in outliers]
    
    QC_metrics[('corr','Physio (resp)')] = df_resp
    QC_metrics[('cov','Physio (resp)')] = df_resp
else:
    QC_metrics[('corr','Physio (resp)')] = None
    QC_metrics[('cov','Physio (resp)')] = None


# ***
# # 8. Load Global Signal Kappa and Rho

# In[21]:


if DATASET == 'evaluation':
    kappa_rho_df = pd.read_csv(f'./summary_files/{DATASET}_gs_kappa_rho.csv', index_col=[0,1])
    print("++ INFO: The shape of kappa_rho_df is %s" % str(kappa_rho_df.shape))
else:
    kappa_rho_df = None


# ***
# # 9. Load GS Timeseries

# In[22]:


gs_df_path = f'./summary_files/{DATASET}_GS_info_and_ts.pkl'
print('++ GS will be loaded from  %s' % gs_df_path)
with open(gs_df_path, 'rb') as f:
    gs_df_dict = pickle.load(f)


# # 9. Load ICA Timeseries and basic statistics

# In[23]:


get_ipython().run_cell_magic('time', '', "ica_ts_sf_path = f'./summary_files/{DATASET}_ICAs.pkl'\nprint('++ INFO: Loading ICA TS (and some extras) from  %s' % ica_ts_sf_path)\nwith open(ica_ts_sf_path, 'rb') as f:\n    ica_dict = pickle.load(f)\n")


# # 10. Load Physiological Recordings

# In[24]:


get_ipython().run_cell_magic('time', '', "physio_ts_sf_path = f'./summary_files/{DATASET}_Physiological_Timeseries.pkl'\nprint('++ INFO: Physiological Recordings will be loaded from: %s' % physio_ts_sf_path)\nwith open(physio_ts_sf_path, 'rb') as f:\n    physio_dict = pickle.load(f)\n")


# # 12. Load Physiological Regressors

# In[25]:


get_ipython().run_cell_magic('time', '', "physio_regressors_sf_path = f'./summary_files/{DATASET}_Physiological_Regressors.pkl'\nprint('++ INFO: Physiological Regressors will be loaded from %s' % physio_regressors_sf_path)\nwith open(physio_regressors_sf_path, 'rb') as f:\n    physio_reg_dict = pickle.load(f)\n")


# ***
# 
# # Create Dashboard

# In[26]:


label_mapping = {r:r.replace('ALL_','') for r in pp_opts.values()}


# In[27]:


avial_qc_metrics = list(set([key[1] for key in QC_metrics.keys()]))

sbj_select                              = pn.widgets.Select(name='Subject',        options=sbj_list, width=200)
ses_select                              = pn.widgets.Select(name='Data Type',      options=ses_list+['all'], width=200)
pp_select                               = pn.widgets.Select(name='Pre-processing', options=pp_opts, width=200)
nordic_select                           = pn.widgets.Select(name='m-NORDIC',         options=nordic_opts, width=200)
fc_select                               = pn.widgets.Select(name='FC Metric',      options={'Correlation':'corr','Covariance':'cov'}, width=200)
plot_select                             = pn.widgets.Select(name='Plot type',      options={'Scatter Plot':'scatter','Hex Bin':'hexbin',
                                                                                            'Timeseries':'timeseries',
                                                                                            'FC Matrices across pipelines':'FCmats',
                                                                                            'FC Matrices across echoes':'FCmats_echoes',
                                                                                            'Group Results (Static)':'group_res_static', 'Group Results (Dynamic)':'group_res_dynamic'})

scat_lim_input                          = pn.widgets.FloatInput(name='Scatter Limit Value', value=1., step=0.1, start=0., end=50., width=200)
show_line_fit_checkbox                  = pn.widgets.Toggle(name='Show Linear Fit', button_type='primary')
scatter_extra_confs_card                = pn.Card(scat_lim_input,show_line_fit_checkbox, title='Scatter Plot & FCs | Configuration')

num_ics_to_show_input                   = pn.widgets.IntInput(name='Number of ICAs', value=3, width=200)
ts_confs_card                           = pn.Card(num_ics_to_show_input, title='Timeseries | Configuration')

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
           scatter_extra_confs_card,pn.layout.Divider(),ts_confs_card,pn.layout.Divider(),
           barplot_extra_confs_card,
           ]


# In[28]:


@pn.depends(sbj_select,ses_select, pp_select, nordic_select, fc_select, plot_select, show_line_fit_checkbox, scat_lim_input,show_stats_toggle,stat_test_select,annot_type_select,pps_to_include_in_group_results,remove_outliers_from_swarm_plots_toggle, qc_metric_select,show_points_toggle,num_ics_to_show_input)
def get_main_frame(sbj,ses, pp, nordic, fc_metric, plot_type, show_line_fit, ax_lim,show_stats,stat_test,annot_type,pps_to_include_in_barplot,remove_outliers_from_swarm_plots, qc_metric,show_points,num_ics_to_show):
    if plot_type == 'timeseries':
        frame = get_ts_report_page(sbj,ses,pp,nordic,fc_metric,num_ics_to_show,
                       gs_df_dict,ica_dict,physio_dict,physio_reg_dict,pBOLD_xr,QC_metrics)
        return frame
    if plot_type == 'hexbin':
        frame = fc_across_echoes_scatter_page(DATASET,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, 
                                              data_fc,pBOLD_xr,QC_metrics,
                                              show_line=show_line_fit, ax_lim=ax_lim, hexbin=True)
        return frame
    if plot_type == 'scatter':
        frame = fc_across_echoes_scatter_page(DATASET,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, 
                                              data_fc,pBOLD_xr,QC_metrics,
                                              show_line=show_line_fit, ax_lim=ax_lim, hexbin=False)
        #frame = fc_across_echoes_scatter_page(DATASET,data_fc,pBOLD_xr,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, hexbin=False)
        return frame
    if plot_type == 'FCmats':
        fcR = get_fc_matrices(data_fc,pBOLD_xr,sbj,ses, nordic, 'corr', net_cmap=power264_nw_cmap)
        fcC = get_fc_matrices(data_fc,pBOLD_xr,sbj,ses, nordic, 'cov', net_cmap=power264_nw_cmap)
        return pn.Column(fcR,fcC)
    if plot_type == 'FCmats_echoes':
        layout = pn.GridBox(ncols=3)
        for ep in echo_pairs:
            fcR = get_fc_matrix(data_fc,pBOLD_xr,sbj,ses,pp,nordic,fc_metric,echo_pair=ep, net_cmap=power264_nw_cmap, ax_lim=ax_lim, title='%s | %s' % (fc_metric,ep))
            layout.append(fcR)
        return layout
    if plot_type == 'group_res_static':
        #data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        if ses == 'all':
            data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        else:
            idx = pd.IndexSlice
            data_to_show = QC_metrics[fc_metric,qc_metric].set_index(['Pre-processing','Session']).loc[idx[pps_to_include_in_barplot, ses], :].reset_index()
        aa = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  hue='Pre-processing',x='m-NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left',show_points=show_points, session=ses, dot_size=1),    title=f'{qc_metric} grouped by m-NORDIC')
        bb = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  x='Pre-processing',hue='m-NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left', show_points=show_points, session=ses, dot_size=1),   title=f'{qc_metric} grouped by Pre-processing')
        return pn.GridBox(*[aa,bb],ncols=2)
    if plot_type == 'group_res_dynamic':
        if fc_metric == 'cov':
            pBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, fc_metric, 'pBOLD', nordic),title='pBOLD')
            return pn.Row(pBOLD_card,None,None)
        else:
            return pn.pane.Markdown('# This is not available for R-based FC')


# In[29]:


template = pn.template.BootstrapTemplate(title=f'{DATASET} Dataset | Edge-based Results', 
                                         sidebar=sidebar,
                                         main=get_main_frame)


# In[30]:


dashboard = template.show() #template.show(port=port_tunnel, open=False)


# In[42]:


dashboard.stop()


# ***
# 
# Here are a few snapshots of the dashboard
# 
# ![Scatter Plots](figures/pBOLD_Dashboard_ScatterPlots.png)
# 
# ***
# 
# ![Timeseries](figures/pBOLD_Dashboard_Timeseries.png)
# 
# ***
# 
# ![Static Reports](figures/pBOLD_Dashboard_StaticReports.png)
# 
# ***
# 
# 
# ![FC Matrices](figures/pBOLD_Dashboard_FCMatrices.png)
# 
# 

# 
