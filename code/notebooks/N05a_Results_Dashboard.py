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
#port_tunnel = 45719

import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import pickle
import panel as pn
from nilearn.connectome import sym_matrix_to_vec

# +
from utils.basics import compute_residuals, echo_pairs, pairs_of_echo_pairs, echo_pairs_tuples, get_dataset_index
from utils.basics import TES_MSEC
from utils.basics import ATLASES_DIR, PRCS_DATA_DIR, PRJ_DIR, FMRI_FINAL_NUM_SAMPLES, FMRI_TRS, NUM_DISCARDED_VOLUMES, DOWNLOAD_DIRS
from utils.basics import mse_dist, chord_distance_between_intersecting_lines, read_group_physio_reports
from utils.dashboard import fc_across_echoes_scatter_page, get_fc_matrices, get_static_report, get_fc_matrix,dynamic_summary_plot_gated, get_ts_report_page

from sklearn.ensemble import IsolationForest


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

CENSORING_MODE='ALL'

fMRI_Sampling_Rate      = FMRI_TRS[DATASET]
fMRI_Preproc_Nsamples   = FMRI_FINAL_NUM_SAMPLES[DATASET]
fMRI_Ndiscard_samples   = NUM_DISCARDED_VOLUMES[DATASET]
fMRI_Preproc_Start_Time = str(float(fMRI_Sampling_Rate.replace('s','')) * int(fMRI_Ndiscard_samples))+'s'
print('++ fMRI Sampling Rate            = %s' % fMRI_Sampling_Rate)
print('++ fMRI Final Number of Samples  = %d Acquisitions' % fMRI_Preproc_Nsamples)
print('++ fMRI Number discarded Samples = %d Acquisitions' % fMRI_Ndiscard_samples)
print('++ fMRI Preproc Start Time       = %s' % fMRI_Preproc_Start_Time)

DOWNLOAD_DIR = DOWNLOAD_DIRS[DATASET]

echo_times_dict = TES_MSEC[DATASET]
print(echo_times_dict)

ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())

fMRI_Preproc_index = pd.timedelta_range(start=fMRI_Preproc_Start_Time, periods=fMRI_Preproc_Nsamples, freq=fMRI_Sampling_Rate)
fMRI_Preproc_index[0:3]

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
if CENSORING_MODE == 'ALL':
    pp_opts = {'No Censoring | No Regression':f'{CENSORING_MODE}_NoRegression',
           'No Censoring | Basic':f'{CENSORING_MODE}_Basic',
           'No Censoring | GSR':f'{CENSORING_MODE}_GS',
           'No Censoring | Tedana (fastica)':f'{CENSORING_MODE}_Tedana-fastica'}
else:
    pp_opts = {'Censoring | No Regression':f'{CENSORING_MODE}_NoRegression',
               'Censoring | Basic':f'{CENSORING_MODE}_Basic',
           'Censoring | GSR':f'{CENSORING_MODE}_GS',
           'Censoring | Tedana (fastica)':f'{CENSORING_MODE}_Tedana-fastica'}
nordic_opts = {'Do not use':'off', 'Active':'on'}

data_fc = {}
# -

# %%time
filename = f'./cache/{DATASET}_fc_{CENSORING_MODE}.pkl'
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
filename = f'./cache/{DATASET}_pBOLD_all_scatters_{CENSORING_MODE}.nc'
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
    scat_plot_weights.loc[ppe] = chord_distance_between_intersecting_lines(1.0, this_case_BOLD_slope, r=0.5)

scat_plot_weights.to_dataframe(name='chord').sort_values(by='chord',ascending=False)

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

# ***
# # 6. Tedana derived metrics

# +
# %%time
for tedana_metric in ['#ICs (All)','#ICs (Likely BOLD)','#ICs (Unlikely BOLD)','Var. Exp. (Likely BOLD)','Var. Exp. (Unlikely BOLD)']:
    aux_df = pd.DataFrame(columns=['Subject','Session','Pre-processing','NORDIC',tedana_metric])
    aux_df.set_index(['Subject','Session','Pre-processing','NORDIC'], inplace=True)
    QC_metrics['C',tedana_metric] = aux_df
    
for sbj in tqdm(sbj_list):
    for ses in ses_list:
        partial_key = (sbj, ses)
        sbj_ses_in_fc = any(key[:len(partial_key)] == partial_key for key in data_fc)
        if not sbj_ses_in_fc:
            print('++ WARNING: This combination of sbj,ses [%s,%s] is not available. XR will contain np.nan.' % (sbj,ses))
            continue
        for nordic in nordic_opts.values():
            for pp in pp_opts.values():
                if 'Tedana' not in pp:
                    QC_metrics['C','#ICs (All)'].loc[sbj,ses,pp,nordic]                = np.nan
                    QC_metrics['C','#ICs (Likely BOLD)'].loc[sbj,ses,pp,nordic]        = np.nan
                    QC_metrics['C','#ICs (Unlikely BOLD)'].loc[sbj,ses,pp,nordic]      = np.nan
                    QC_metrics['C','Var. Exp. (Likely BOLD)'].loc[sbj,ses,pp,nordic]   = np.nan
                    QC_metrics['C','Var. Exp. (Unlikely BOLD)'].loc[sbj,ses,pp,nordic] = np.nan
                else:
                    tedana_type = pp.split(f'{CENSORING_MODE}_Tedana-')[1]
                    d_folder    = f'D03_Preproc_{ses}_NORDIC-{nordic}'
                    ica_metrics_path = osp.join(PRCS_DATA_DIR,sbj,d_folder,f'tedana_{tedana_type}','ica_metrics.tsv')
                    ica_metrics              = pd.read_csv(ica_metrics_path, sep='\t').set_index('Component')
                    likely_bold_components   = list(ica_metrics[ica_metrics['classification_tags']=='Likely BOLD'].index)
                    unlikely_bold_components = list(ica_metrics[ica_metrics['classification_tags']=='Unlikely BOLD'].index)
                    
                    QC_metrics['C','#ICs (All)'].loc[sbj,ses,pp,nordic]              = ica_metrics.shape[0]
                    QC_metrics['C','#ICs (Likely BOLD)'].loc[sbj,ses,pp,nordic]      = len(likely_bold_components)
                    QC_metrics['C','#ICs (Unlikely BOLD)'].loc[sbj,ses,pp,nordic]    = len(unlikely_bold_components)
                    QC_metrics['C','Var. Exp. (Likely BOLD)'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[likely_bold_components,'variance explained'].sum().round(2)
                    QC_metrics['C','Var. Exp. (Unlikely BOLD)'].loc[sbj,ses,pp,nordic] = ica_metrics.loc[unlikely_bold_components,'variance explained'].sum().round(2)
# -

# ***
# # 7. Physiological Recording Derived Metrics

if DATASET == 'evaluation':
    report_card_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_card_review_all_scans.txt')
    report_card_summary_df    = read_group_physio_reports(report_card_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_card_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_card = report_card_summary_df['peak ival over dset mean std'].copy()
    df_card.columns=['Mean','St.Dev.']
    df_card['color'] = ['red' if c else 'green' for c in outliers]
    
    QC_metrics[('R','Physio (cardiac)')] = df_card
    QC_metrics[('C','Physio (cardiac)')] = df_card
else:
    QC_metrics[('R','Physio (cardiac)')] = None
    QC_metrics[('C','Physio (cardiac)')] = None

if DATASET == 'evaluation':
    report_resp_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_resp_review_all_scans.txt')
    report_resp_summary_df    = read_group_physio_reports(report_resp_summary_path)
    
    clf    = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(report_resp_summary_df['peak ival over dset mean std'])
    outliers = labels == -1
    df_resp = report_resp_summary_df['peak ival over dset mean std'].copy()
    df_resp.columns=['Mean','St.Dev.']
    df_resp['color'] = ['red' if c else 'green' for c in outliers]
    
    QC_metrics[('R','Physio (resp)')] = df_resp
    QC_metrics[('C','Physio (resp)')] = df_resp
else:
    QC_metrics[('R','Physio (resp)')] = None
    QC_metrics[('C','Physio (resp)')] = None

for tedana_metric in ['#ICs (All)','#ICs (Likely BOLD)','#ICs (Unlikely BOLD)','Var. Exp. (Likely BOLD)','Var. Exp. (Unlikely BOLD)']:
    QC_metrics['C',tedana_metric] = QC_metrics['C',tedana_metric].reset_index()
    QC_metrics['R',tedana_metric] = QC_metrics['C',tedana_metric].copy()

import pickle 
with open(f'./cache/{DATASET}_QC_metrics_{CENSORING_MODE}.pkl', 'wb') as f:
    pickle.dump(QC_metrics, f)

# # 8. Load the Global Signal Timeseries

if DATASET == 'evaluation':
    kappa_rho_df = pd.read_csv(f'./cache/{DATASET}_gs_kappa_rho.{CENSORING_MODE}.csv', index_col=[0,1])
    print("++ INFO: The shape of kappa_rho_df is %s" % str(kappa_rho_df.shape))
else:
    kappa_rho_df = None

# %%time
filename = f'./cache/{DATASET}_{CENSORING_MODE}_GS_info_and_ts.pkl'
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    with open(filename, 'rb') as f:
        gs_df_dict = pickle.load(f)
else:
    gs_df_dict = {}
    for sbj,ses in tqdm(ds_index):
        if DATASET == 'evaluation':
            # Load All the GS versions (each echo time and optimally combined)
            gs_e01_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e01.volreg.GS.1D')
            gs_e02_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.1D')
            gs_e03_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e03.volreg.GS.1D')
            gs_OC_path  = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb06.{sbj}.r01.tedana_fastica_OC.GS.1D')
            gs_e01 = np.loadtxt(gs_e01_path)
            gs_e02 = np.loadtxt(gs_e02_path)
            gs_e03 = np.loadtxt(gs_e03_path)
            gs_OC  = np.loadtxt(gs_OC_path)
            gs_df = pd.DataFrame([gs_OC,gs_e01,gs_e02,gs_e03],index=['OC','TE1','TE2','TE3']).T
            gs_df = gs_df.infer_objects()
            gs_df.index = fMRI_Preproc_index
            gs_df.index.name = 'Time'
            gs_df.columns.name='Echo Time'
            # Transform to units of Signal Percent Change
            gs_df_spc = 100*(gs_df-gs_df.mean())/gs_df.mean()
            gs_df_dict[(sbj,ses,'gs_ts')] = gs_df_spc
    
            # Create Dataframe with Gs properties of interest
            GS_phys_match_file = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
            try:
                with open(GS_phys_match_file, 'rb') as f:
                    loaded_dict = pickle.load(f)
                gs_adjr2_physio = float(loaded_dict['model'].rsquared_adj)
            except:
                gs_adjr2_physio = None
            gs_kappa        = float(kappa_rho_df.loc[(sbj,ses),'kappa (GS)'])
            gs_rho          = float(kappa_rho_df.loc[(sbj,ses),'rho (GS)'])
            gs_df_metrics   = pd.DataFrame([gs_adjr2_physio,gs_kappa,gs_rho],index=['Adj R2 Physio','kappa','rho'],columns=['GS']).T
            gs_df_dict[sbj,ses,'gs_metrics'] = gs_df_metrics
        else:
            gs_df_dict[(sbj,ses,'gs_ts')] = None
            gs_df_dict[sbj,ses,'gs_metrics'] = None
    with open(filename, 'wb') as f:
        pickle.dump(gs_df_dict, f)

# # 9. Load ICA Timeseries and basic statistics

# %%time
filename = f'./cache/{DATASET}_{CENSORING_MODE}_ICAs.pkl'
print(filename)
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    with open(filename, 'rb') as f:
        ica_dict = pickle.load(f)
else:
    ica_dict = {}
    for sbj,ses in tqdm(ds_index):
        # Load IC Timeseries
        ic_ts_path         = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off','tedana_fastica',f'ica_mixing.tsv')
        ic_ts              = pd.read_csv(ic_ts_path, sep='\t')
        ic_ts              = ic_ts.infer_objects()
        ic_ts.index        = fMRI_Preproc_index
        ic_ts.index.name   = 'Time'
        ic_ts.columns.name = 'Components'
        ica_dict[(sbj,ses,'ic_ts')] = ic_ts
        # Load IC Properties
        ic_metrics = pd.read_csv(osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off','tedana_fastica',f'ica_metrics.tsv'),sep='\t', index_col=0)
        ic_metrics.index.name='Name'
        ic_metrics               = ic_metrics.round(2)[['kappa','rho','variance explained','classification_tags']]
        if DATASET == 'evaluation':
            ic_metrics['corrwithGS'] = ic_ts.corrwith(gs_df_dict[sbj,ses,'gs_ts']['OC'])
            ic_metrics.columns = ['kappa','rho','varepx','label','R(ic,GS)']
        else:
            ic_metrics['corrwithGS'] = np.nan
            ic_metrics.columns = ['kappa','rho','varepx','label','R(ic,GS)']
        ica_dict[(sbj,ses,'ic_metrics')] = ic_metrics
    with open(filename, 'wb') as f:
        pickle.dump(ica_dict, f)

# # 10. Load Physiological Recordings

from afnipy import lib_physio_reading as lpr
from afnipy import lib_physio_opts    as lpo
import copy

# %%time
filename = f'./cache/{DATASET}_{CENSORING_MODE}_Physiological_Timeseries.pkl'
print(filename)
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    with open(filename, 'rb') as f:
        physio_dict = pickle.load(f)
else:
    physio_dict = {}
    for sbj,ses in tqdm(ds_index):
        if DATASET == 'evaluation':
            slibase_file = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio',f'{sbj}_{ses}_task-rest_echo-1_slibase.1D')
            if osp.exists(slibase_file):
                # Load Physio by creating Afni RetroObj
                phys_file = osp.join(DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.tsv.gz')
                json_file = osp.join(DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.json')
                dset_epi  = osp.join(DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_echo-1_bold.nii.gz')
                input_line = ['./physio_calc.py', '-phys_file', phys_file, '-phys_json', json_file, '-dset_epi', dset_epi, 
                              '-prefilt_mode', 'median', '-prefilt_max_freq', '50', '-verb','0']
                args_orig  = copy.deepcopy(input_line)
                args_dict  = lpo.main_option_processing( input_line )
                retobj     = lpr.retro_obj( args_dict, args_orig=args_orig )
                physio_start_time = retobj.start_time
                # Extract Cardiac Timseries
                card_end_time  = retobj.data['card'].end_time
                card_nsamples  = retobj.data['card'].ts_orig.shape[0]
                card_samp_rate = retobj.data['card'].samp_rate
                card_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=card_nsamples, freq=str(card_samp_rate)+"s")
                card_df        = pd.DataFrame(retobj.data['card'].ts_orig, index=card_index,columns=['PGG'])
                card_df.index.name = 'Time'
    
                # Extract Respiratory Timeseries
                resp_end_time  = retobj.data['resp'].end_time
                resp_nsamples  = retobj.data['resp'].ts_orig.shape[0]
                resp_samp_rate = retobj.data['resp'].samp_rate
                resp_index     = pd.timedelta_range(start=str(physio_start_time)+"s", periods=resp_nsamples, freq=str(resp_samp_rate)+"s")
                resp_df        = pd.DataFrame(retobj.data['resp'].ts_orig, index=resp_index,columns=['Respiration'])
                resp_df.index.name = 'Time'
    
                #Add to final dictionary
                physio_dict[(sbj,ses,'card')] = card_df
                physio_dict[(sbj,ses,'resp')] = resp_df
            else:
                physio_dict[(sbj,ses,'card')] = None
                physio_dict[(sbj,ses,'resp')] = None
        else:
            physio_dict[(sbj,ses,'card')] = None
            physio_dict[(sbj,ses,'resp')] = None
    with open(filename, 'wb') as f:
        pickle.dump(physio_dict, f)

# # 12. Load Physiological Regressors

from afnipy.lib_afni1D import Afni1D

# %%time
filename = f'./cache/{DATASET}_{CENSORING_MODE}_Physiological_Regressors.pkl'
print(filename)
if osp.exists(filename):
    print("++ WARNING: Loading pre-existing data from cache folder.")
    with open(filename, 'rb') as f:
        physio_reg_dict = pickle.load(f)
else:
    physio_reg_dict = {}
    for sbj,ses in tqdm(ds_index):
        if DATASET == 'evaluation':
            slibase_path = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio',f'{sbj}_{ses}_task-rest_echo-1_slibase.1D')
            if osp.exists(slibase_path):
                slibase_obj  = Afni1D(slibase_path)
                slibase_df   = pd.read_csv(slibase_path, comment='#', delimiter=' +', header=None, engine='python')
                slibase_df.columns=slibase_obj.labels
                slibase_df=slibase_df[3::].reset_index(drop=True)
                slibase_df.index = fMRI_Preproc_index
                slibase_df.index.name = 'Time'
                slibase_df.columns.name = 'Regressors'
                # Load list of selected regressors for the GS
                GS_phys_match_file = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
                with open(GS_phys_match_file, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    selected_physio_regs = loaded_dict['selected_regs']
                selected_RVT_regs  = [r for r in selected_physio_regs if 'rvt' in r]
                selected_card_regs = [r for r in selected_physio_regs if 'card' in r]
                selected_resp_regs = [r for r in selected_physio_regs if 'resp' in r]
                physio_reg_dict[(sbj,ses,'RVT_regs')]  = slibase_df[selected_RVT_regs]
                physio_reg_dict[(sbj,ses,'card_regs')] = slibase_df[selected_card_regs]
                physio_reg_dict[(sbj,ses,'resp_regs')] = slibase_df[selected_resp_regs]
            else:
                physio_reg_dict[(sbj,ses,'RVT_regs')] = None
                physio_reg_dict[(sbj,ses,'card_regs')] = None
                physio_reg_dict[(sbj,ses,'resp_regs')] = None
        else:
            physio_reg_dict[(sbj,ses,'RVT_regs')] = None
            physio_reg_dict[(sbj,ses,'card_regs')] = None
            physio_reg_dict[(sbj,ses,'resp_regs')] = None
    with open(filename, 'wb') as f:
        pickle.dump(physio_reg_dict, f)

# ***
#
# # Create Dashboard

label_mapping = {r:r.replace(f'{CENSORING_MODE}_','') for r in pp_opts.values()}
label_mapping

# +
avial_qc_metrics = list(set([key[1] for key in QC_metrics.keys()]))

sbj_select                              = pn.widgets.Select(name='Subject',        options=sbj_list, width=200)
ses_select                              = pn.widgets.Select(name='Data Type',      options=ses_list+['all'], width=200)
pp_select                               = pn.widgets.Select(name='Pre-processing', options=pp_opts, width=200)
nordic_select                           = pn.widgets.Select(name='NORDIC',         options=nordic_opts, width=200)
fc_select                               = pn.widgets.Select(name='FC Metric',      options={'Correlation':'R','Covariance':'C'}, width=200)
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


# -

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
        #data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        if ses == 'all':
            data_to_show = QC_metrics[fc_metric,qc_metric].set_index('Pre-processing').loc[pps_to_include_in_barplot].reset_index()
        else:
            idx = pd.IndexSlice
            data_to_show = QC_metrics[fc_metric,qc_metric].set_index(['Pre-processing','Session']).loc[idx[pps_to_include_in_barplot, ses], :].reset_index()
        aa = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  hue='Pre-processing',x='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left',show_points=show_points, session=ses, dot_size=1),    title=f'{qc_metric} grouped by NORDIC')
        bb = pn.Card(get_static_report(data_to_show,fc_metric,qc_metric,  x='Pre-processing',hue='NORDIC',
                                stat_test=stat_test, show_stats=show_stats, stat_annot_type=annot_type, 
                                 remove_outliers_from_swarm=remove_outliers_from_swarm_plots, legend_location='lower left', show_points=show_points, session=ses, dot_size=1),   title=f'{qc_metric} grouped by Pre-processing')
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

from utils.dashboard import gen_scatter

sbj,ses='sub-20','ses-2'
gen_scatter(DATASET,data_fc,sbj,ses,'ALL_Basic','off','e01|e02','e03|e03','C', show_linear_fit=False, ax_lim=None, hexbin=False, title=None, color='red') *\
gen_scatter(DATASET,data_fc,sbj,ses,'ALL_Basic','on','e01|e02','e03|e03','C', show_linear_fit=False, ax_lim=None, hexbin=False, title=None, color='green')

import seaborn as sns
import matplotlib.pyplot as plt

# +
sbj,ses='sub-20','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))

# +
sbj,ses='sub-16','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))

# +
sbj,ses='sub-31','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))

# +
sbj,ses='sub-95','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
#ax.set_xlim(data_df3['e01|e01'].quantile(0.01),data_df3['e01|e01'].quantile(0.99))
#ax.set_ylim(data_df3['e03|e03'].quantile(0.01),data_df3['e03|e03'].quantile(0.99))

# +
sbj,ses='sub-16','ses-1'
data_df = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e01|e01','C'].values, discard_diagonal=True)
data_df['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','off','e03|e03','C'].values, discard_diagonal=True)
data_df['NORDIC']='off'
data_df2 = pd.DataFrame(columns=['e01|e01','e03|e03', 'NORDIC'])
data_df2['e01|e01'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e01|e01','C'].values, discard_diagonal=True)
data_df2['e03|e03'] = sym_matrix_to_vec(data_fc[sbj,ses,'ALL_Basic','on','e03|e03','C'].values, discard_diagonal=True)
data_df2['NORDIC']='on'
data_df3=pd.concat([data_df,data_df2])

fig,ax = plt.subplots(1,1)
sns.kdeplot(data=data_df3,x='e01|e01',y='e03|e03',hue='NORDIC',ax=ax, fill=True, alpha=0.5)
# -









import holoviews as hv
def gen_scatter(dataset,data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_linear_fit=False, ax_lim=None, hexbin=False, title=None, color='blue'):
    """
    Generate scatter plot for two different FC matrices

    Inputs:
    -------
    dataset (str): name of the dataset being used (e.g., evaluation, discovery)
    data_fc (dict): dictionary with FC matrices
    sbj (str): subject ID
    ses (str): session ID
    pp (str): pre-processing pipeline
    nordic (str): whether Nordic was used or not
    eep1 (str): first echo pair in the format 'e02|e02'
    eep2 (str): second echo pair in the format 'e02|e02'
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    show_linear_fit (bool): whether to show linear fit line or not
    ax_lim (float): axis limits for the scatter plot
    hexbin (bool): whether to use hexbin plot instead of scatter plot

    Returns:
    --------
    hvplot object with the scatter plot and theoretical lines

    """
    echo_times_dict = TES_MSEC[dataset]

    if (sbj,ses,pp,nordic,eep1,fc_metric) not in data_fc:
        return pn.pane.Markdown('#Not Available')
    data_df = pd.DataFrame(columns=[eep1,eep2, fc_metric])
    data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)
    data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)
    # Compute limits for X and Y axis
    if ax_lim is None:
        if fc_metric == 'R':
            lims = (-1,1) 
        else:
            lims = (data_df.quantile(0.01).min(),data_df.quantile(0.99).max())
    else:
        lims=(-ax_lim,ax_lim)
    # Create scatter plot and fitted line
    scat           = data_df.hvplot.scatter(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7, title=title, color=color) #.opts(active_tools=['save'], tools=['save'])
    data_lin_fit   = hv.Slope.from_scatter(scat).opts(line_width=3, line_color='#0f0fff') #.opts(active_tools=['save'], tools=['save'])

    if hexbin:
        scat = data_df.hvplot.hexbin(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7)
    # Compute theoretical slopes for extreme BOLD and So dominated regimes
    if fc_metric  == 'R':
        BOLD_line_sl, BOLD_line_int = 1.,0.
    else:
        e1_X,e2_X     = eep1.split('|')
        e1_Y,e2_Y     = eep2.split('|')
        BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])
        BOLD_line_int = 0.
    
    So_line_sl, So_line_int = 1.,0.
    
    # Create Theoretical BOLD and So lines
    BOLD_line  = hv.Slope(BOLD_line_sl, BOLD_line_int).opts(line_color='g',line_width=2, line_dash='dashed') #.opts(active_tools=['save'], tools=['save'])
    So_line    = hv.Slope(So_line_sl,   So_line_int  ).opts(line_color='r',line_width=2, line_dash='dashed') #.opts(active_tools=['save'], tools=['save'])
    
    # Join all graphical elements
    if show_linear_fit:
        plot = (scat * data_lin_fit * So_line * BOLD_line) #.opts(toolbar=None)
    else:
        plot = (scat * So_line * BOLD_line)
     
    return plot.opts(active_tools=['reset'])
























# +
import holoviews as hv
data_qc = QC_metrics
resp_plot, card_plot = None,None
if (sbj,ses) in data_qc['C',('Physio (resp)')].index:
    resp_plot = data_qc['C',('Physio (resp)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Card. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
    resp_plot = resp_plot * hv.Points((data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
    resp_plot = resp_plot.opts(shared_axes=False)
if (sbj,ses) in data_qc['C',('Physio (cardiac)')].index:
    card_plot = data_qc['C',('Physio (cardiac)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Resp. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
    card_plot = card_plot * hv.Points((data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
    card_plot = card_plot.opts(shared_axes=False)

physio_card = pn.Card(pn.Column(resp_plot,card_plot),title='Physio',width=350)
physio_card
# -

accuracy_dict['Pearson'].set_index(['Session']).loc['ses-1'].reset_index(drop=True)

dashboard.stop()

# ***
#
# # Explore extreme cases
# ## a) Worse pBOLD scan

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(2, 100))
motion_df = pd.DataFrame(index=ds_index, columns = ['Max. Motion (enorm)','Mean Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max. Motion (enorm)'] = aux_mot.max()
motion_df['Mean Motion (dot size)'] = mms.fit_transform(motion_df['Mean Motion (enorm)'].values.reshape(-1,1))
motion_df['Max. Motion (dot size)'] = mms.fit_transform(motion_df['Max. Motion (enorm)'].values.reshape(-1,1))
motion_df = motion_df.infer_objects()
motion_val = motion_df.loc[sbj,ses].values[0]

pBOLD_df = QC_metrics['C','pBOLD']
pBOLD_df = pBOLD_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)

TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)



# +
df = pd.concat([TSNR_df.set_index(['Subject','Session']),pBOLD_df.set_index(['Subject','Session']), motion_df],axis=1)
cbar_min = df['Max. Motion (enorm)'].quantile(0.05)
cbar_max = df['Max. Motion (enorm)'].quantile(0.99)

df.hvplot.scatter(y='TSNR (Full Brain)',x='pBOLD', hover_cols=['Subject','Session'],aspect='square',s='Max. Motion (dot size)',c='Max. Motion (enorm)', cmap='cividis', fontscale=1.5, frame_width=350).opts(clim=(cbar_min,cbar_max),colorbar_opts={'title':'Max. Motion (mm):'})
# -

df

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set seaborn theme
sns.set(style="white", context="notebook")

# Compute quantile limits for color normalization
vmin = df["Max. Motion (enorm)"].quantile(0.05)
vmax = df["Max. Motion (enorm)"].quantile(0.99)

# Create JointGrid
g = sns.JointGrid(data=df, x="pBOLD", y="TSNR (Full Brain)", space=0, height=8)

# Scatter plot with color and size
df = df[["pBOLD", "TSNR (Full Brain)", "Max. Motion (dot size)", "Max. Motion (enorm)"]].dropna()

#sizes = df["Max. Motion (dot size)"]
#scaled_sizes = 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min()) + 10  # Scale between 10 and 210
sc = g.ax_joint.scatter(
    data=df,
    x="pBOLD",
    y="TSNR (Full Brain)",
    s=df['Max. Motion (dot size)'],
    c=df["Max. Motion (enorm)"],
    cmap="cividis",
    vmin=vmin, vmax=vmax,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.3
)

# Create new axes above the top marginal histogram
cbar_ax = g.fig.add_axes([pos_joint.x0, pos_marg_x.y1 + 0.05, pos_joint.width, 0.02])  # [left, bottom, width, height]

# Add horizontal colorbar
cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')

# Move label and ticks to the top
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.tick_top()
cbar.set_label("Max. Motion (mm)", rotation=0, labelpad=5)

# Top and right marginal plots
sns.histplot(x=df["pBOLD"], ax=g.ax_marg_x, kde=True, color="#72B6A1", edgecolor="black", bins=50)
sns.histplot(y=df["TSNR (Full Brain)"], ax=g.ax_marg_y, kde=True, color="#72B6A1", edgecolor="black", bins=50)

# Axis labels
g.set_axis_labels("$p_{BOLD}$", "TSNR (Full Brain)")

# Improve layout
plt.tight_layout()
plt.show()


# -

df.info()





sbj,ses,pBOLD_val = pBOLD_df.reset_index(drop=True).sort_values(by='pBOLD',ascending=True).iloc[0]
TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
TSNR_val = TSNR_df.set_index(['Subject','Session']).loc[sbj,ses].values[0]
(sbj,ses)

motion_df = pd.DataFrame(index=ds_index, columns = ['Max Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        #motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_val = motion_df.loc[sbj,ses].values[0]

# +
fig,ax = plt.subplots(1,3, figsize=(10,5))
sns.barplot(data=pBOLD_df, alpha=0.5,ax=ax[0])
sns.swarmplot(data=pBOLD_df,s=3,ax=ax[0])
ax[0].axhline(pBOLD_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=TSNR_df, alpha=0.5,ax=ax[1])
sns.swarmplot(data=TSNR_df,s=2,ax=ax[1])
ax[1].axhline(TSNR_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=motion_df, ax=ax[2], alpha=0.5)
sns.swarmplot(data=motion_df,s=2,ax=ax[2])
ax[2].axhline(motion_val, color='k',linestyle='dashed', linewidth=0.5)
# -

# ## b) Second Worse TSNR scan

import seaborn as sns
import matplotlib.pyplot as plt

pBOLD_df = QC_metrics['C','pBOLD']
pBOLD_df = pBOLD_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
sbj,ses,pBOLD_val =pBOLD_df.reset_index(drop=True).sort_values(by='pBOLD',ascending=True).iloc[1]
TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
TSNR_val = TSNR_df.set_index(['Subject','Session']).loc[sbj,ses].values[0]
(sbj,ses)

# +
fig,ax = plt.subplots(1,3, figsize=(10,5))
sns.barplot(data=pBOLD_df, alpha=0.5,ax=ax[0])
sns.swarmplot(data=pBOLD_df,s=3,ax=ax[0])
ax[0].axhline(pBOLD_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=TSNR_df, alpha=0.5,ax=ax[1])
sns.swarmplot(data=TSNR_df,s=2,ax=ax[1])
ax[1].axhline(TSNR_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=motion_df, ax=ax[2], alpha=0.5)
sns.swarmplot(data=motion_df,s=2,ax=ax[2])
ax[2].axhline(motion_val, color='k',linestyle='dashed', linewidth=0.5)
# -

# ## b) Third Worse TSNR scan

import seaborn as sns
import matplotlib.pyplot as plt

pBOLD_df = QC_metrics['C','pBOLD']
pBOLD_df = pBOLD_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
sbj,ses,pBOLD_val =pBOLD_df.reset_index(drop=True).sort_values(by='pBOLD',ascending=True).iloc[2]
TSNR_df = QC_metrics['C','TSNR (Full Brain)']
TSNR_df = TSNR_df.set_index(['Pre-processing','NORDIC']).loc[f'{CENSORING_MODE}_Basic','off'].dropna().reset_index(drop=True)
TSNR_val = TSNR_df.set_index(['Subject','Session']).loc[sbj,ses].values[0]
(sbj,ses)

motion_df = pd.DataFrame(index=ds_index, columns = ['Max Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        #motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_val = motion_df.loc[sbj,ses].values[0]

# +
fig,ax = plt.subplots(1,3, figsize=(10,5))
sns.barplot(data=pBOLD_df, alpha=0.5,ax=ax[0])
sns.swarmplot(data=pBOLD_df,s=3,ax=ax[0])
ax[0].axhline(pBOLD_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=TSNR_df, alpha=0.5,ax=ax[1])
sns.swarmplot(data=TSNR_df,s=2,ax=ax[1])
ax[1].axhline(TSNR_val, color='k',linestyle='dashed', linewidth=0.5)

sns.barplot(data=motion_df, ax=ax[2], alpha=0.5)
sns.swarmplot(data=motion_df,s=2,ax=ax[2])
ax[2].axhline(motion_val, color='k',linestyle='dashed', linewidth=0.5)
# -

motion_df


