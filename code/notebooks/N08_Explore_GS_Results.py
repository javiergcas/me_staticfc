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

import os.path as osp
import pandas as pd
import numpy as np
import os
import copy
from tqdm import tqdm
import datetime
from utils.basics import PRJ_DIR, PRCS_DATA_DIR, SPRENG_DOWNLOAD_DIR, CODE_DIR,NUM_DISCARDED_VOLUMES
from utils.basics import detrend_signal, get_dataset_index
import hvplot.pandas
import holoviews as hv
import xarray as xr
from utils.dashboard import get_static_report
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy.stats import zscore
from bokeh.models.formatters import DatetimeTickFormatter
formatter = DatetimeTickFormatter(minutes = '%Mmin:%Ssec')
from afnipy.lib_afni1D import Afni1D
from afnipy import lib_physio_reading as lpr
from afnipy import lib_physio_opts as lpo

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

DATASET='evaluation'

ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())

fMRI_num_discarded_volumes = NUM_DISCARDED_VOLUMES['evaluation']

# ***
# # 1. Load data
#
# ### 1.1. Kappa and Rho for Global Signal

kappa_rho_df = pd.read_csv('./cache/gs_kappa_rho.csv', index_col=[0,1])
print("++ INFO: The shape of kappa_rho_df is %s" % str(kappa_rho_df.shape))
kappa_rho_df.head(2)

# ### 1.2. Load Variance Explained by Physiological Regressors
#
# > **NOTE:** This dataframe contains less entries than the one above because good physio was not available for all scans.

real_varex_df = pd.read_csv('./cache/real_varexp_gs_physio.csv', index_col=[0,1])
print("++ INFO: The shape of real_varex_df is %s" % str(real_varex_df.shape))
real_varex_df.head(2)

# We also load the null distribution of variance explained in GS by physio regressors

null_varex_df = pd.read_csv('./cache/null_varexp_gs_physio.csv', index_col=[0])
print("++ INFO: The shape of null_varex_df is %s" % str(null_varex_df.shape))
null_varex_df.head(2)

# ### 1.3. Load Head Motion estimates

mms = MinMaxScaler(feature_range=(2, 100))
motion_df = pd.DataFrame(index=ds_index, columns = ['Mean Motion (enorm)','Max Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_df['Mean Motion (dot size)'] = mms.fit_transform(motion_df['Mean Motion (enorm)'].values.reshape(-1,1))
motion_df['Max Motion (dot size)'] = mms.fit_transform(motion_df['Max Motion (enorm)'].values.reshape(-1,1))

# ### 1.4. Load TSNR and pBOLD for all scans

import pickle
with open(f'./cache/{DATASET}_QC_metrics.pkl', 'rb') as f:
    QC_metrics = pickle.load(f)
QC_metrics.keys()

# ***
# # 2. Explore GS Kappa and Rho
#
# ### 2.1. Is GS Kappa (i.e., BOLD) or Rho (i.e., non-BOLD) dominated?
#
# Below we show a scatter plot of Kappa vs. Rho:
#
# 1. Each dot represents a scan
# 2. The dotted line is the 45o (identity) line.
# 3. Any scan above the 45o line has kappa < rho (marked in red)
# 4. Any scan below the 45o line has kappa > rho (marked in green)
# 5. Dots size is proportional to mean motion (estimated via enorm)
#

df = pd.concat([kappa_rho_df, motion_df], axis=1)
df.head(3)

kappa_vs_rho_byType = df.hvplot.scatter(x='kappa (GS)',y='rho (GS)', aspect='square', hover_cols=['Subject','Session'], c='kappa_rho_color',s='Mean Motion (dot size)', title='Global Signal Kappa vs. Rho') *hv.Slope(1,0).opts(line_width=0.5,line_dash='dashed', line_color='black')

kappa_vs_rho_byMotion = df.hvplot.scatter(x='kappa (GS)',y='rho (GS)', aspect='square', hover_cols=['Subject','Session'], c='Max Motion (enorm)', title='Global Signal Kappa vs. Rho', cmap='cividis') *hv.Slope(1,0).opts(line_width=0.5,line_dash='dashed', line_color='black')

kappa_vs_rho_byType + kappa_vs_rho_byMotion

# ### 2.2. Does TSNR and pBOLD for GSR behavior depends on whether or not GS is primary BOLD? 
#
# For this, we first create two scan lists:
#
# * ```higher_kappa_scans```: scans whose GS is BOLD dominant (kappa > rho)
# * ```higher_rho_scans```: scans whose GS is non-BOLD dominant (rho > kappa) 

# The cell below shows us how many scans we have in each subgroup.

df['kappa_rho_color'].value_counts()

higher_kappa_scans = kappa_rho_df[kappa_rho_df['kappa_rho_color']=='lightgreen'].index
higher_rho_scans   = kappa_rho_df[kappa_rho_df['kappa_rho_color']=='red'].index

top50_kappa_scans = kappa_rho_df.sort_values(by='kappa (GS)', ascending=False)[0:50].index
bot50_kappa_scans  = kappa_rho_df.sort_values(by='kappa (GS)')[0:50].index

layout=pn.Row()
for qc_metric in ['TSNR (Full Brain)','pBOLD']:    
    df = QC_metrics['C',qc_metric].set_index(['Pre-processing','NORDIC'])
    df = df[df.index.get_level_values('Pre-processing').isin(['ALL_Basic','ALL_GS']) & (df.index.get_level_values('NORDIC') == 'off')]
    df = df.reset_index().set_index(['Subject','Session']).drop(['NORDIC'],axis=1)
    df['Type'] = 'N/A'
    df.loc[higher_kappa_scans,'Type'] = 'BOLD Dominated'
    df.loc[higher_rho_scans,'Type'] = 'So Dominated'
    df = df.replace({'ALL_Basic':'Basic','ALL_GS':'GSR'})
    df = df.dropna()
    plot = get_static_report(df,'C',qc_metric,  hue='Pre-processing',x='Type',session='all', show_points=True, remove_outliers_from_swarm=False, show_stats=True, stat_test='Mann-Whitney', remove_outliers_from_statistics=False, dot_size=2) 
    layout.append(plot)
layout = pn.Column('# Results considering all scans, and subdividing by the 45 degree line',layout)
layout

layout=pn.Row()
for qc_metric in ['TSNR (Full Brain)','pBOLD']:    
    df = QC_metrics['C',qc_metric].set_index(['Pre-processing','NORDIC'])
    df = df[df.index.get_level_values('Pre-processing').isin(['ALL_Basic','ALL_GS']) & (df.index.get_level_values('NORDIC') == 'off')]
    df = df.reset_index().set_index(['Subject','Session']).drop(['NORDIC'],axis=1)
    df['Type'] = 'N/A'
    df.loc[top50_kappa_scans,'Type'] = 'Top 50 Kappa'
    df.loc[bot50_kappa_scans,'Type'] = 'Bottom 50 Kappa'
    df = df[df['Type']!='N/A']
    df = df.replace({'ALL_Basic':'Basic','ALL_GS':'GSR'})
    df = df.dropna()
    plot = get_static_report(df,'C',qc_metric,  hue='Pre-processing',x='Type',session='all', show_points=True, remove_outliers_from_swarm=True, show_stats=True, stat_test='Mann-Whitney', remove_outliers_from_statistics=True, dot_size=2) 
    layout.append(plot)
layout = pn.Column('# Results only top 50 (higher kappa or rho)',layout)
layout

# # 3. Visualize the GS as a carpet plot for all scans

gs_df = pd.DataFrame(index=ds_index, columns=range(201))
for sbj,ses in tqdm(ds_index):
    gs_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb06.{sbj}.r01.tedana_fastica_OC.GS.demean.1D')
    gs = np.loadtxt(gs_path)
    gs_df.loc[(sbj,ses)] = gs

gs_df.loc[df.sort_values(by='rho (GS)', ascending=False).index].reset_index(drop=True).hvplot.heatmap(cmap='gray').opts(clim=(gs_df.melt()['value'].quantile(0.05),gs_df.melt()['value'].quantile(0.95)),width=1000)

# Now, let's plot TNSR and pBOLD for both Basis Regressors and GSR separately for the set of scans whose global signal in BOLD dominant (kappa > rho), and non-BOLD dominant (rho > kappa)

# ***
# # 3. Explore Variance Explained by Physiological Regressors
#
# ### 3.1. How much variance do physio regressors explained in our sample?

non_parametric_p005 = null_varex_df.quantile(0.95).values[0]

hv.VSpan(non_parametric_p005,1.0).opts(fill_color='gray') * \
real_varex_df.hvplot.hist(bins=20, xlim=(0,1), title='Variance Explained by Physio Regressors in the Global Signal',                     ylabel=' % Scans', normed=True, label='Real Data') * \
real_varex_df.hvplot.kde(          xlim=(0,1), title='Variance Explained by Physio Regressors in the Global Signal',                     ylabel=' % Scans',              label='Real Data') * \
hv.Text(0.09,7,'Not Significant') * hv.Text(0.5,7,'Significant') + \
null_varex_df.hvplot.hist(bins=20, xlim=(0,1), title='Variance Explained by Physio Regressors in the Global Signal (NULL Distribution)', ylabel=' % Scans', normed=True, label='Null Distribution', color='gray') * \
null_varex_df.hvplot.kde(          xlim=(0,1), title='Variance Explained by Physio Regressors in the Global Signal',                     ylabel=' % Scans',              label='Null Distribution', color='gray')

num_scans_with_physio_not_significantly_explaining_any_variance = (real_varex_df<non_parametric_p005).sum().values[0]
pc_scans_with_physio_not_significantly_explaining_any_variance = 100 * num_scans_with_physio_not_significantly_explaining_any_variance / real_varex_df.shape[0]
print("++ INFO: Number and [percentage] of scans for which physio regressors DO NOT explain a signficiant amount of variance: %d [ %.2f%% ]" % (num_scans_with_physio_not_significantly_explaining_any_variance,pc_scans_with_physio_not_significantly_explaining_any_variance))





# So, for only 45% of scans we can say that physiological regressors explain a significant amount of variance in the GS. Moreover, of these the range of explained variances if from 18% to 55%, which most scans sitting on the lower end of this range.
#
# My current interpretation is that becuase GS does not seem to be mostly explaianble by physiological BOLD, then its removal might not be a good thing (as noted by pBOLD)

# ### 3.2. Let's explore in a bit more detail
#
# Next we plot (for all scans) the varexp by physio as the hight of bars that are sorted in the X-axis by the GS kappa. We do not see any clear trend, which suggest that the kappaness of the GS is not solely explained by the amount of variance of the physiological regressors. There is some other form of BOLD fluctuation (that does not relate to physio regressors) that adds additional BOLDness to the GS on a maner that is scan dependent.

df=pd.concat([motion_df,kappa_rho_df,real_varex_df],axis=1).dropna()
for c in df.columns:
    new_c = c + ' RANK'
    df[new_c] = df[c].rank(ascending=False)
df.head(2)

df.hvplot.bar(x='kappa (GS) RANK',y='Var. Exp. by Physio Regressors', width=1000)

df.hvplot.bar(x='rho (GS) RANK',y='Mean Motion (enorm)', width=1000)

# ### 3.2. Let's explore in detail the scan with the GS being best explained by physio regressors
#
# Let's find the scan with the highest variance explained by physio regressors in the global signal

(sbj_top_varexp,ses_top_varex) = df[df['Var. Exp. by Physio Regressors RANK'] == 1.0].index[0]
df[df['Var. Exp. by Physio Regressors RANK'] == 1.0]

# Let's find the scan with the lowest variance explained by physio regressors in the global signal

(sbj_bot_varexp,ses_bot_varex) = df[df['Var. Exp. by Physio Regressors RANK'] == df.shape[0]].index[0]
df[df['Var. Exp. by Physio Regressors RANK'] == df.shape[0]]

# Let's find the scan with the GS that has the highest Kappa

(sbj_top_kappa,ses_top_kappa) = df[df['kappa (GS) RANK'] == 1.0].index[0]
df[df['kappa (GS) RANK'] == 1.0]

# Let's find the scan with the GS that has the lowest Kappa

(sbj_bot_kappa,ses_bot_kappa) = df[df['kappa (GS) RANK'] == df.shape[0]].index[0]
df[df['kappa (GS) RANK'] == df.shape[0]]

scans_to_show = {'Highest Physio Var. Exp.':(sbj_top_varexp,ses_top_varex),
                 'Lowest Physio Var. Exp.':(sbj_bot_varexp,ses_bot_varex),
                 'Highest Kappa': (sbj_top_kappa,ses_top_kappa),
                 'Lowest Kappa': (sbj_bot_kappa,ses_bot_kappa)}

signals = {}
for scan_type, (sbj, ses) in scans_to_show.items():
    # Create Paths of interest
    gs_path        = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.1D')
    model_path     = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'pb03.{sbj}.r01.e02.volreg.GS.PhysioModeling.pkl')
    slibase_path   = osp.join(PRCS_DATA_DIR,sbj,'D04_Physio',f'{sbj}_{ses}_task-rest_echo-2_slibase.1D')
    phys_file_path = osp.join(SPRENG_DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.tsv.gz')
    phys_json_path = osp.join(SPRENG_DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.json')
    dset_path      = osp.join(SPRENG_DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_echo-2_bold.nii.gz')
    # Create container for this paricular scan
    signals[(sbj,ses)] = {}
    # Load Global signal
    signals[(sbj,ses)]['GS'] = pd.DataFrame(np.loadtxt(gs_path),columns=['GS (original)'])
    signals[(sbj,ses)]['GS']['GS (detrended)'] = detrend_signal(signals[(sbj,ses)]['GS']['GS (original)'])
    # Load Predicted global signal
    model = sm.load(model_path)
    signals[(sbj,ses)]['GS']['Predicted GS (by physio)'] = model.predict()

    # Load Physio Data
    argv = ['NA', '-phys_file', phys_file_path, '-phys_json', phys_json_path, '-dset_epi', dset_path]
    args_orig = copy.deepcopy(argv)
    args_dict = lpo.main_option_processing(argv)
    retobj    = lpr.retro_obj( args_dict, args_orig=args_orig, verb=0)

    fmri_tr   = retobj.vol_tr
    fmri_nt   = retobj.vol_nv - fMRI_num_discarded_volumes
    resp_sr   = retobj.data['resp'].samp_rate
    card_sr   = retobj.data['card'].samp_rate

    resp_tcut_samples = int(fMRI_num_discarded_volumes * fmri_tr / resp_sr)
    resp_nsamples     = retobj.data['resp'].ts_orig.shape[0] - resp_tcut_samples
    resp_ts           = retobj.data['resp'].ts_orig[resp_tcut_samples:]
    signals[(sbj,ses)]['Resp'] = pd.DataFrame(resp_ts,index=pd.to_timedelta(np.arange(resp_nsamples)*resp_sr,'s'))

    card_tcut_samples = int(fMRI_num_discarded_volumes * fmri_tr / card_sr)
    card_nsamples     = retobj.data['card'].ts_orig.shape[0] - card_tcut_samples
    card_ts           = retobj.data['card'].ts_orig[card_tcut_samples:]
    signals[(sbj,ses)]['Card'] = pd.DataFrame(card_ts,index=pd.to_timedelta(np.arange(card_nsamples)*card_sr,'s'))

    # Update the index for GS now that we know the fMRI TR
    signals[(sbj,ses)]['GS'].index = pd.to_timedelta(np.arange(fmri_nt)*fmri_tr,'s')

    # Load Physiological Regressors
    slibase_obj        = Afni1D(slibase_path)
    slibase_df         = pd.read_csv(slibase_path, comment='#', delimiter=' +', header=None, engine='python')
    slibase_df.columns = slibase_obj.labels
    slibase_df         = slibase_df[int(fMRI_num_discarded_volumes)::].reset_index(drop=True)
    slibase_df.index   = pd.to_timedelta(np.arange(fmri_nt)*fmri_tr,'s')
    slibase_det_df = slibase_df.copy()
    for c in slibase_det_df.columns:
        slibase_det_df[c] = detrend_signal(slibase_df[c].values.squeeze())
    signals[(sbj,ses)]['Slibase'] = slibase_det_df
    phys_reg_list = np.unique([c.split('.',1)[1] for c in slibase_det_df.columns])

scan_select = pn.widgets.Select(name='Representative scan', options=scans_to_show)
sidebar     = [scan_select]


# +
@pn.depends(scan_select)
def get_main_frame(scan):
    sbj,ses = scan
    p1 = pn.layout.Card(show_gs_and_prediction(sbj,ses),title='Predicted Global Signal by Physiological Regressors')
    p2 = pn.layout.Card(show_gs_and_respiration(sbj,ses),title='Global Signal and Respiration')
    p3 = pn.layout.Card(show_gs_and_cardiac(sbj,ses),title='Global Signal and Cardiac Recording')
    return pn.Column(p1,p2,p3)
def show_gs_and_prediction(sbj,ses):
    """
    Plot the GS and the prediction based on physiological regressors"""
    data = signals[(sbj,ses)]['GS'].apply(zscore)
    rsquared  = np.power(np.corrcoef(data['GS (detrended)'],data['Predicted GS (by physio)'])[0,1],2)*100
    data.columns.name='Data:'
    
    plot = data.hvplot(y=['GS (detrended)','Predicted GS (by physio)'], width=1500, line_width=[5,1],line_color=['blue','blue'], line_dash=['solid','dashed'], ylabel='Normalized Signal Units', xlabel='Scan Time', xformatter=formatter, ylim=(-3,3.5))
    plot = plot * hv.Text(data.index[100].to_timedelta64(),3,'   Variance Explained = %.2f %%' % rsquared)
    plot.opts(legend_cols=2, legend_position='bottom_left')
    return plot

def show_gs_and_respiration(sbj,ses):
    """
    Plot the GS in conjunction with respiration recording and RVT
    """
    gs_df      = signals[(sbj,ses)]['GS'].apply(zscore)
    resp_df    = signals[(sbj,ses)]['Resp'].apply(zscore)
    slibase_df = signals[(sbj,ses)]['Slibase'].apply(zscore)
    resp_plot = resp_df.hvplot(width=1500, label='Respiratory Signal').opts(line_color='gray')
    gs_plot   = gs_df.hvplot(y='GS (detrended)', xlabel='Time', ylabel='Normalized Signal Units', label='GS (detrended)').opts(line_width=5,line_color='blue')
    rvt_plot  = slibase_df.hvplot(y='s000.rvt00', label='RVT').opts(line_color='k')

    plot = resp_plot * gs_plot * rvt_plot
    plot = plot.opts(legend_cols=3, legend_position='bottom_left', title='GS and Respiration for scan with highest GS Kappa')
    
    return plot

def show_gs_and_cardiac(sbj,ses):
    """
    Plot the GS in conjunction with cardiac recording
    """
    gs_df      = signals[(sbj,ses)]['GS'].apply(zscore)
    card_df    = signals[(sbj,ses)]['Card'].apply(zscore)
    slibase_df = signals[(sbj,ses)]['Slibase'].apply(zscore)
    card_plot   = card_df.hvplot(width=1500, label='Cardiac Signal').opts(line_color='gray')
    gs_plot     = gs_df.hvplot(y='GS (detrended)', xlabel='Time', ylabel='Normalized Signal Units', label='GS (detrended)').opts(line_width=5,line_color='blue')
    cardc1_plot = slibase_df.hvplot(y='s000.card.c1', label='Cardiac (c1)').opts(line_color='g')
    cards1_plot = slibase_df.hvplot(y='s000.card.c2', label='Cardiac (s1)').opts(line_color='y')

    plot = card_plot * gs_plot * cardc1_plot * cards1_plot
    plot = plot.opts(legend_cols=3, legend_position='bottom_left', title='GS and Cardiac signals for scan with highest GS Kappa')
    
    return plot


# +
template = pn.template.BootstrapTemplate(title='Global Signal Dashboard | Representative Scans', 
                                         sidebar=sidebar,
                                         main=get_main_frame)

dashboard = template.show(port=port_tunnel)
# -

dashboard.stop()
