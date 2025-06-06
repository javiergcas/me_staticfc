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

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

from utils.basics import TES_MSEC, SESSIONS
from utils.basics import ATLASES_DIR, PRCS_DATA_DIR, PRJ_DIR
import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import panel as pn
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from utils.basics import compute_residuals, softmax, echo_pairs, echo_pairs_tuples, pairs_of_echo_pairs
from utils.dashboard import fc_across_echoes_scatter_page, get_fc_matrices, get_barplot, get_fc_matrix,dynamic_summary_plot_gated, get_barplot
import pickle


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


# ***

echo_times_dict = TES_MSEC['Spreng_Scanner1']
ses_list        = SESSIONS['Spreng_Scanner1']

ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])

# # 1. Basic Information
# Create lists with all 6 possible echo combinations, and then all possible pairings between those.

print('Echo Pairs[n=%d] = %s' %(len(echo_pairs),str(echo_pairs)))
print('Pairs of Echo Pairs[n=%d] = %s' %(len(pairs_of_echo_pairs),str(pairs_of_echo_pairs)))

echo_times_dict

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

# Create a dictionary to be used as colormap when plotting FC matrices

power264_nw_cmap = {nw:roi_info_df.set_index('Network').loc[nw]['RGB'].values[0] for nw in list(roi_info_df['Network'].unique())}

# ***
# # 3. Load Timeseries and compute R and C matrices
#
# This cell will load ROI Timeseries, compute R and C, and place these into a dictionary of datafrmes. It will do this for the Basic denoising pipeline (Basic) and no censoring (ALL).

# +
pp_opts = {'No Censoring | Basic':'ALL_Basic',
           'No Censoring | GSR':'ALL_GSasis',
           'No Censoring | Tedana':'ALL_Tedana', 
           'No Censoring | Tedana (Nc=88)':'ALL_Tedana-NORDIC_FixNComps'}
nordic_opts = {'Do not use':'Off', 'Active':'On'}

data_fc = {}
# -

# %%time
filename = './cache/spreng_fc.pkl'
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    with open(filename, 'rb') as f:
        data_fc = pickle.load(f)
else:
    for sbj,ses in tqdm(list(dataset_info_df.index)):
        for nordic in nordic_opts.values():
            for pp in pp_opts.values():
                for (e_x,e_y) in echo_pairs_tuples:
                    # Compose Dfolder name
                    if (nordic == 'Off') and ('NORDIC_FixNComps' not in pp):
                        d_folder = f'D02_Preproc_fMRI_{ses}'
                    elif (nordic == 'On')  and ('NORDIC_FixNComps' not in pp):
                        d_folder = f'D04_Preproc_fMRI_{ses}_NORDIC'
                    elif ('NORDIC_FixNComps' in pp):
                        d_folder = f'D05_Preproc_fMRI_{ses}_NORDIC_FixNComps'
                    pp_suffix = pp.replace('-NORDIC_FixNComps','') # Will need to be removed once I re-organize files <=========================================
                    # Compose path to input TS
                    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{pp_suffix}.{ATLAS_NAME}_000.netts')
                    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{pp_suffix}.{ATLAS_NAME}_000.netts')
                    # Load TS into memory
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

# ***
#
# # 4. Compute QA-metrics

sbj_list  = sorted(list((set(dataset_info_df.index.get_level_values(level='Subject')))))

# %%time
filename = './cache/spreng_fc_qc.nc'
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    qa_xr = xr.open_dataarray(filename)
else:
    qa_xr = xr.DataArray(dims=['sbj','ses','pp','nordic','fc_metric','ee_vs_ee','qc_metric',],
                         coords={'sbj':sbj_list,
                                 'ses':['ses-1','ses-2'],
                                 'pp': list(pp_opts.values()),
                                 'nordic':list(nordic_opts.values()),
                                 'fc_metric':['R','C'],
                                 'ee_vs_ee':pairs_of_echo_pairs,
                                 'qc_metric':['dBOLD','dSo','pBOLD','pSo','TSNR (Full Brain)','TSNR (Visual Cortex)']})
    for sbj in tqdm(sbj_list):
        for ses in  ['ses-1','ses-2']:
            partial_key = (sbj, ses)
            sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)
            if not sbj_ses_in_fc:
                print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))
                continue
            for fc_metric in ['R','C']:
                for pp in pp_opts.values():
                    for nordic in nordic_opts.values():
                        for eep in pairs_of_echo_pairs:
                            # Extract vectorized FC for this particular case
                            eep1,eep2 = eep.split('_vs_')
                            data_df = pd.DataFrame(columns=[eep1,eep2])
                            data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)
                            data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)
    
                            # Calculate slope and intercept for the two extreme scenarios
                            So_line_sl, So_line_int = 1.,0. # This is always the same
                            BOLD_line_int = 0.              # This is always the same
                            if fc_metric  == 'R':
                               BOLD_line_sl = 1.
                            if fc_metric == 'C':
                                e1_X,e2_X     = eep1.split('|')
                                e1_Y,e2_Y     = eep2.split('|')
                                BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])
                            # Compute dBOLD and dSo metrics
                            qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'dBOLD'] = np.sqrt((compute_residuals(data_df[eep1].values,data_df[eep2].values,BOLD_line_sl,BOLD_line_int)**2).sum())
                            qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'dSo']   = np.sqrt((compute_residuals(data_df[eep1].values,data_df[eep2].values,So_line_sl,  So_line_int)**2).sum())
                            # REMOVE OUTLIERS: qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dBOLD'] = np.sqrt((reject_outliers(compute_residuals(data_df[eep1].values,data_df[eep2].values,BOLD_line_sl,BOLD_line_int),3)**2).sum())
                            # REMOVE OUTLIERS: qa_xr.loc[sbj,ses,pp,nordic,'C',eep,'dSo']   = np.sqrt((reject_outliers(compute_residuals(data_df[eep1].values,data_df[eep2].values,So_line_sl,  So_line_int),3)**2).sum())
                            # Compute probabilities
                            qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'pBOLD'], qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'pSo'] = 1 - softmax(qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,['dBOLD','dSo']].values)

                            #TSNR 
                            if nordic == 'Off' :
                                d_folder = f'D02_Preproc_fMRI_{ses}'
                            elif (nordic == 'On') & (pp=='ALL_Tedana-NORDIC_FixNComps'):
                                d_folder = f'D05_Preproc_fMRI_{ses}_NORDIC_FixNComps'
                            else:
                                d_folder = f'D04_Preproc_fMRI_{ses}_NORDIC'

                            pp_suffix = pp.replace('-NORDIC_FixNComps','')
                            aux_rois_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tsnr_stats_regress',f'TSNR_ROIs_e02_{pp_suffix}.txt')
                            aux_fb_path   = osp.join(PRCS_DATA_DIR,sbj,d_folder,'tsnr_stats_regress',f'TSNR_FB_e02_{pp_suffix}.txt')
                            if osp.exists(aux_rois_path) and osp.exists(aux_fb_path):
                                aux_rois = pd.read_csv(aux_rois_path,skiprows=3, sep='\s+').drop(0).set_index('ROI_name')
                                aux_fb   = pd.read_csv(aux_fb_path,skiprows=3, sep='\s+').drop(0).set_index('ROI_name')
                                qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'TSNR (Visual Cortex)'] = float(aux_rois.loc['GHCP-R_Primary_Visual_Cortex','Tmed'])
                                qa_xr.loc[sbj,ses,pp,nordic,fc_metric,eep,'TSNR (Full Brain)']    = float(aux_rois.loc['GHCP-R_Primary_Visual_Cortex','Tmed'])
                            else:
                                print('++ WARNING: TSNR info missing for [%s,%s,%s]' % (sbj,ses,aux_rois_path))
    qa_xr.to_netcdf(filename)

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
plot_select   = pn.widgets.Select(name='Plot type',      options={'Scatter Plot':'scatter','Hex Bin':'hexbin','FC Matrices across pipelines':'FCmats',
                                                                 'FC Matrices across echoes':'FCmats_echoes',
                                                                 'Group Results (Static)':'group_res_static', 'Group Results (Dynamic)':'group_res_dynamic'})

scat_lim_input                          = pn.widgets.FloatInput(name='Scatter Limit Value', value=1., step=0.1, start=0., end=50., width=200)
show_line_fit_checkbox                  = pn.widgets.Toggle(name='Show Linear Fit', button_type='primary')
scatter_extra_confs_card                = pn.Card(scat_lim_input,show_line_fit_checkbox, title='Scatter Plot & FCs | Configuration')
pps_to_include_in_group_results         = pn.widgets.MultiSelect(name='Pipelines to include in Group Results', options=pp_opts, value=list(pp_opts.values())[0:3], width=200)
remove_outliers_from_swarm_plots_toggle = pn.widgets.Toggle(name='Remove Outliers from BarPlot', button_type='primary')

show_stats_toggle = pn.widgets.Toggle(name='Show Statistical Annotations', button_type='primary')
stat_test_select  = pn.widgets.Select(name='Statistical Test', options={'Paired T-test':'t-test_paired','Independent T-test':'t-test_ind','Mann Whitney (Ind,non-param)':'Mann-Whitney'})
annot_type_select = pn.widgets.Select(name='Annotation Type', options={'Stars':'star','Simple Annotation':'simple','Full Annotation':'full'})
barplot_extra_confs_card = pn.Card(show_stats_toggle,stat_test_select,annot_type_select,pps_to_include_in_group_results,remove_outliers_from_swarm_plots_toggle, title='Group Results (Static) | Configuration')
sidebar = [sbj_select,ses_select,pp_select,nordic_select,fc_select, pn.layout.Divider(),
           plot_select,pn.layout.Divider(),
           scatter_extra_confs_card,pn.layout.Divider(),
           barplot_extra_confs_card,
           ]


# -

@pn.depends(sbj_select,ses_select, pp_select, nordic_select, fc_select, plot_select, show_line_fit_checkbox, scat_lim_input,show_stats_toggle,stat_test_select,annot_type_select,pps_to_include_in_group_results,remove_outliers_from_swarm_plots_toggle)
def get_main_frame(sbj,ses, pp, nordic, fc_metric, plot_type, show_line_fit, ax_lim,show_stats,stat_test,annot_type,pps_to_include_in_barplot,remove_outliers_from_swarm_plots):
    if plot_type == 'hexbin':
        frame = fc_across_echoes_scatter_page(data_fc,qa_xr,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, other_stats=other_stats[nordic], hexbin=True)
        return frame
    if plot_type == 'scatter':
        frame = fc_across_echoes_scatter_page(data_fc,qa_xr,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=show_line_fit, ax_lim=ax_lim, other_stats=other_stats[nordic], hexbin=False)
        return frame
    if plot_type == 'FCmats':
        fcR = get_fc_matrices(data_fc,qa_xr,sbj,ses, nordic, 'R', net_cmap=power264_nw_cmap)
        fcC = get_fc_matrices(data_fc,qa_xr,sbj,ses, nordic, 'C', net_cmap=power264_nw_cmap)
        return pn.Column(fcR,fcC)
    if plot_type == 'FCmats_echoes':
        layout = pn.GridBox(ncols=3)
        for ep in echo_pairs:
            fcR = get_fc_matrix(data_fc,qa_xr,sbj,ses,pp,nordic,fc_metric,echo_pair=ep, net_cmap=power264_nw_cmap, ax_lim=ax_lim, title='%s | %s' % (fc_metric,ep))
            layout.append(fcR)
        return layout
    if plot_type == 'group_res_static':
        a = pn.Card(get_barplot(qa_xr.sel(pp=pps_to_include_in_barplot),nordic,fc_metric,'pBOLD',  hue='Pre-processing',x='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left'),title='pBOLD for Speng Sample (1)')
        b = pn.Card(get_barplot(qa_xr.sel(pp=pps_to_include_in_barplot),nordic,fc_metric,'pBOLD',  x='Pre-processing',hue='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left'),title='pBOLD for Speng Sample (2)')
        c = pn.Card(get_barplot(qa_xr.sel(pp=pps_to_include_in_barplot),nordic,fc_metric,'TSNR (Full Brain)',  hue='Pre-processing',x='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left'),title='TSNR for Speng Sample (1)')
        d = pn.Card(get_barplot(qa_xr.sel(pp=pps_to_include_in_barplot),nordic,fc_metric,'TSNR (Full Brain)',  x='Pre-processing',hue='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left'),title='TSNR for Speng Sample (2)')
        return pn.GridBox(*[a,b,c,d],ncols=2)
        #return pn.Row(get_barplot(qa_xr,nordic,fc_metric,'pBOLD',  hue='Pre-processing',x='NORDIC',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type),
        #              get_barplot(qa_xr,nordic,fc_metric, 'dBOLD', hue='Pre-processing',x='NORDIC',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type),
        #              get_barplot(qa_xr,nordic,fc_metric, 'dSo',   hue='Pre-processing',x='NORDIC',stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type))
    if plot_type == 'group_res_dynamic':
        if fc_metric == 'C':
            pBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, fc_metric, 'pBOLD', nordic),title='pBOLD')
            #dBOLD_card = pn.Card(dynamic_summary_plot_gated(qa_xr, fc_metric, 'dBOLD', nordic),title='dBOLD')
            #dS0_card   = pn.Card(dynamic_summary_plot_gated(qa_xr, fc_metric, 'dSo', nordic),title='dSo')
            return pn.Row(pBOLD_card,None,None)
            #return pn.Row(pBOLD_card,dBOLD_card,dS0_card)
        else:
            return pn.pane.Markdown('# This is not available for R-based FC')


template = pn.template.BootstrapTemplate(title='Evaluation Dataset (Spreng et al.) | Edge-based Results', 
                                         sidebar=sidebar,
                                         main=get_main_frame)

dashboard = template.show(port=port_tunnel)

# ***

dashboard.stop()

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
    


