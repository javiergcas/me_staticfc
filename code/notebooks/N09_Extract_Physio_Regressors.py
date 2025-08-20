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

# # Description: Generation and evaluation of physio regressors contribution to the GS
#
# This notebook will do the following:
#
# 1) Attempt to run AFNI program ```physio_calc``` on scans for which physiological timeseries are available.
# 2) Automaticall detect a subset of scans where ```physio_calc``` has done its job correctly
# 3) Compute the variance explained in the global signal by physiological regressors
# 4) Create a null distribution of variance explained

import os.path as osp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from utils.basics import PRJ_DIR, PRCS_DATA_DIR, SPRENG_DOWNLOAD_DIR, CODE_DIR, read_group_physio_reports
from sklearn.ensemble import IsolationForest
import hvplot.pandas
import holoviews as hv
from scipy.stats import zscore
import statsmodels.api as sm
from afnipy.lib_afni1D import Afni1D
import shutil

import getpass
username = getpass.getuser()
print(username)

# Get list of fMRI scans from the Spreng Dataset that passed our intial QC and are included in this study

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])

# ***
# # 1. Run ```physio_calc``` in all scans with physio data available
#
# We will do this in biowulf using batch jobs. Here we create the necessary infrastructure for that
#
# ### 1.1. Create Swarm path

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N09a_Compute_Physio_Regressors.SWARM.sh')
print(script_path)

# ### 1.2. Create folder for log files

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N09a_Compute_Physio_Regressors.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

# ### 1.3. Write Swarm File
#
# This file will contain one line per scan (sbj,ses) for which we were able to find both ```{sbj}_{ses}_task-rest_physio.tsv.gz``` and ```{sbj}_{ses}_task-rest_physio.json``` files

# +
no_physio_scans = []

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for sbj,ses in tqdm(dataset_info_df.index):
        physio_path = osp.join(SPRENG_DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.tsv.gz')
        json_path   = osp.join(SPRENG_DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_physio.json')
        if osp.exists(physio_path) and osp.exists(json_path):
            for ec in [1,2,3]:
                dset_path = osp.join(SPRENG_DOWNLOAD_DIR,sbj,ses,'func',f'{sbj}_{ses}_task-rest_echo-{ec}_bold.nii.gz')
                out_dir = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio')
                prefix = f'{sbj}_{ses}_task-rest_echo-{ec}'
                #the_file.write(f'physio_calc.py -phys_file {physio_path} -phys_json {json_path} -dset_epi {dset_path} -out_dir {out_dir} -prefix {prefix} \n')
                the_file.write(f'physio_calc.py -phys_file {physio_path} -phys_json {json_path} -dset_epi {dset_path} -out_dir {out_dir} -prefix {prefix} -prefilt_mode median -prefilt_max_freq 50\n')
        else:
            no_physio_scans.append((sbj,ses))    
the_file.close()     
# -

# ### 1.4 Run batch jobs in biowulf
#
# ```bash
#
# swarm -f /data/SFIMJGC_HCP7T/BCBL2024/swarm.javiergc/N09a_Compute_Physio_Regressors.SWARM.sh -g 8 -t 8 -b 20 --time 00:10:00 --logdir /data/SFIMJGC_HCP7T/BCBL2024/logs.javiergc/N09a_Compute_Physio_Regressors.log --partition quick,norm --module afni
# ```

# ***
# # 2. Check the outputs of ```phys_calc```
#
# For some scans, even though the physio files are available, they do not contain a sufficient amount of samples. In those cases ```physio_calc``` cannot run.
#
# We next try to identify those instances and come with a final list of scans for which ```physio_calc``` completed. 
#
# ### 2.1. Get list of scans with existing automatically computed physiological regressors
#
# First, we get the list of scans in which we attempted to run physio_calc

sbj_ses_with_physio = dataset_info_df.drop(no_physio_scans).index

# Second, we check for which of these scans there is an ```slibase``` file. This is the primary output of ```phys_calc``` that contains the RVT and RETROICOR regressors.

sbj_ses_physio_corrupted = []
for sbj,ses in tqdm(sbj_ses_with_physio):
    for ec in [1,2,3]:
        file_path = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio',f'{sbj}_{ses}_task-rest_echo-{ec}_slibase.1D')
        if not osp.exists(file_path):
            sbj_ses_physio_corrupted.append((sbj,ses))

sbj_ses_physio_corrupted = list(set(sbj_ses_physio_corrupted))
print('++ INFO: Number of scans with physio available, but somehow corrupted: %d scans' % len(sbj_ses_physio_corrupted))
print(sbj_ses_physio_corrupted)

# Finally, we create a new list that only contains the scans for which ```physio_calc``` was able to generate a ```slibase``` file.

scans_with_complete_physio = dataset_info_df.drop(no_physio_scans + sbj_ses_physio_corrupted).index
print("++ INFO: Number of scans for which we were able to complete physio_calc = %d scans" % len(scans_with_complete_physio))

# ## 2.2 Check some basic statistics to ensure consistent physiological regressors quality
#
# First, we will use AFNI's ```gen_ss_review_table.py``` to gather into a single file quality metrics regarding peak detection on the cardiac and respiratory traces. As one can expect for cardiac and respiratory rates to be somehow within a limited range, looking for scans where these are outliers is one good way to automatically detect scans with incorrect physiological regressors.
#
# ```bash
# ml afni
# # cd /data/SFIMJGC_HCP7T/BCBL2024/prcs_data
#
# gen_ss_review_table.py -overwrite \
#                        -write_table ./physio_card_review_all_scans.txt \
#                        -infiles ./sub-*/D06_Physio/sub-*_ses-?_task-rest_echo-1_card_review.txt
#
# gen_ss_review_table.py -overwrite \
#                        -write_table ./physio_resp_review_all_scans.txt \
#                        -infiles ./sub-*/D06_Physio/sub-*_ses-?_task-rest_echo-1_resp_review.txt
#                        
# ```

# We now read these pysio report files, one for cardiac regressors and one for respiration regressors

report_card_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_card_review_all_scans.txt')
report_resp_summary_path  = osp.join(PRJ_DIR,'prcs_data','physio_resp_review_all_scans.txt')

# ### 2.3 Detect scans where peak detection for cardiac traces might have failed
#
# To detect potential errors in cardiac peak detection, we will characterize each scan by its mean inter-peak interval and its standard devation. We will then use IsolationForest to detect outliers, namely scans whose mean and standard deviation deviate from the most common ones in the sample

report_card_summary_df = read_group_physio_reports(report_card_summary_path)

clf    = IsolationForest(contamination=0.1, random_state=42)
labels = clf.fit_predict(report_card_summary_df['peak ival over dset mean std'])
outliers = labels == -1
df_card = report_card_summary_df['peak ival over dset mean std'].copy()
df_card.columns=['Mean','St.Dev.']
df_card['color'] = ['red' if c else 'green' for c in outliers]
df_card.hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Cardiac Inter-peak Interval (seconds)', aspect='square', hover_cols=['Subject','Run'], alpha=0.5) * hv.VSpan(60,100)

# In the figure above, each dot represents a scan. Red dots are scans marked as outliers.
#
# Finally, we create a list with the non-outlier scans. These are scans that, from the perspective of peak detection in the cardiac traces, are valid for further analyses.

scans_with_reasonable_cardiac = df_card[df_card['color']=='green'].index
len(scans_with_reasonable_cardiac)

# ### 2.4 Detect scans where peak detection for respiratory traces might have failed
#
# Here we apply the same logic as in the above section, but this time looking at the inter-peak interval statistics for the respiratory traces

report_resp_summary_df = read_group_physio_reports(report_resp_summary_path)

clf    = IsolationForest(contamination=0.1, random_state=42)
labels = clf.fit_predict(report_resp_summary_df['peak ival over dset mean std'])
outliers = labels == -1
df_resp = report_resp_summary_df['peak ival over dset mean std'].copy()
df_resp.columns=['Mean','St.Dev.']
df_resp['color'] = ['red' if c else 'green' for c in outliers]
df_resp.hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Cardiac Inter-peak Interval (seconds)', aspect='square', hover_cols=['Subject','Run'], alpha=0.5) * hv.VSpan(60,100)

# In the figure above, each dot represents a scan. Red dots are scans marked as outliers.
#
# Finally, we create a list with the non-outlier scans. These are scans that, from the perspective of peak detection in the respiratory traces, are valid for further analyses.

scans_with_reasonable_resp = df_resp[df_resp['color']=='green'].index
len(scans_with_reasonable_resp)

# ### 2.5. Combine information gathered from cardiac and respiratory peak detection to create a final list of scans
#
# We now create a final list of scans with valid physiological data by keeping only scans not marked as outliers both from the cardiac and respiration perspective.

selected_scans = scans_with_reasonable_cardiac.intersection(scans_with_reasonable_resp)
print("++ INFO: Number of scans with reasonable physiological regressors: %d scans" % len(selected_scans))

# ***
#
# # 3. Compute variance explained in the GS for physiological regressors
#
# For the scans that we know have good physio regressors, we will now compute how much variance of the global signal can be explained by the physio regressors. This is done via batch jobs that call program ```GS_physio_exp_var```. The way this program estimates variance explained is as follows:
#
# 1. Load provided global signal and physiological regressors into memory
# 2. Remove constant and linear trends from all loaded timeseries separately
# 3. For each regressor type (e.g., rvt01, rtv02, card.c1, etc) it finds the time-shifted version that most strongly correlates with the global signal. At the end of this step, we will have a list of 13 regressors.
# 4. Computes the variance explained by these 13 regressors.
#
# ### 3.1. Create path for Swarm file

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N09b_Compute_varexp_in_GS_by_physio.SWARM.sh')
print(script_path)

# ### 3.2. Create folder for logs

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N09b_Compute_varexp_in_GS_by_physio.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

# ### 3.3. Write Swarm file
#
# This will contain one line per-scan that we have marked as having good physiological data.

with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for sbj,ses in tqdm(selected_scans):
        gs_path      = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e02.volreg.scale.GSasis.1D')
        slibase_path = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio',f'{sbj}_{ses}_task-rest_echo-2_slibase.1D')
        output_path  = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e02.volreg.scale.GSasis.PhysioModeling.pkl')
        the_file.write(f'export GS_PATH={gs_path} PHYSIO_PATH={slibase_path} OUTPUT_PATH={output_path}; sh {CODE_DIR}/python/GS_physio_exp_var.sh\n')
the_file.close()     

# The next cell help us look for issues when running the batch jobs. If all things went well there should be no WARNING lines printed out.

for sbj,ses in tqdm(selected_scans):
    output_path  = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e02.volreg.scale.GSasis.PhysioModeling.pkl')
    if not osp.exists(output_path):
        print("++ WARNING: %s is missing" % output_path)

# We will now compile all results into a single csv file for later exploration

df = pd.DataFrame(index=selected_scans,columns=['Var. Exp. by Physio Regressors'])
for sbj,ses in tqdm(selected_scans):
    # Variance Explained by Regressors
    model_path  = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e02.volreg.scale.GSasis.PhysioModeling.pkl')
    model = sm.load(model_path)
    df.loc[(sbj,ses),'Var. Exp. by Physio Regressors'] = model.rsquared
df=df.infer_objects()
df.to_csv('./cache/real_varexp_gs_physio.csv')

# ***
#
# # 4. Create a NULL DISTRIBUTION for estimates of variance explained.
#
# Both the global signal and the physiological regressors have quite constrained spectral characteristics. Moreover, to ensure we do not understimate how much physiology can explain the global signal, we are picking the best time-shited version of each regressor. Although those are good things to make sure we do not understimate the contribution of cardiac and respiratory function to the global signal, it can lead to over estimation. By generating a null distribution were we compute the variance explain in the global signal of one scan by the physiological regressors of another scan, we build a null distribution so that we can better contextualize our variance explained estimates.
#
# ### 4.1. Create path for swarm file

script_path = osp.join(PRJ_DIR,f'swarm.{username}',f'N09c_Compute_varexp_in_GS_by_physio_nulls.SWARM.sh')
print(script_path)

# ### 4.2. Create folder for logs

log_path = osp.join(PRJ_DIR,f'logs.{username}',f'N09c_Compute_varexp_in_GS_by_physio_nulls.log')
if not osp.exists(log_path):
    os.makedirs(log_path)
print(log_path)

# ### 4.3. Create a folder where to save the results of each of the 10,000 null permutations

perm_dir = osp.join(CODE_DIR,'notebooks','cache','gs_phys_varex_perms')
if osp.exists(perm_dir):
    shutil.rmtree(perm_dir)
os.makedirs(perm_dir)

# ### 4.4. Write the Swarm file
#
# Here, for each permutation, we first randomly select one scan (sbj,ses) for the global signal. Then we randomly select one scan from any other subject for the physiological regressors.

n_null_cases = 10000
selected_scans_df = pd.DataFrame(index=selected_scans)
with open(script_path, 'w') as the_file:
    the_file.write('# Script Creation Date: %s\n' % str(datetime.date.today()))
    the_file.write(f'# swarm -f {script_path} -g 8 -t 8 -b 20 --time 00:10:00 --logdir {log_path} --partition quick,norm --module afni \n')
    the_file.write('\n')
    for i in tqdm(range(n_null_cases)):
        ii = str(i).zfill(5)
        gs_sbj, gs_ses = selected_scans_df.sample(1).index.values[0]
        ph_sbj, ph_ses = pd.DataFrame(index=selected_scans.drop(gs_sbj,level='Subject')).sample(1).index.values[0]
        gs_path        = osp.join(PRCS_DATA_DIR,gs_sbj,f'D02_Preproc_fMRI_{gs_ses}',f'pb03.{gs_sbj}.r01.e02.volreg.scale.GSasis.1D')
        ph_path        = osp.join(PRCS_DATA_DIR,ph_sbj,'D06_Physio',f'{ph_sbj}_{ph_ses}_task-rest_echo-2_slibase.1D')
        out_path       = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
        the_file.write(f'export GS_PATH={gs_path} PHYSIO_PATH={ph_path} OUTPUT_PATH={out_path}; sh {CODE_DIR}/python/GS_physio_exp_var.sh\n')
the_file.close()     

# ### 4.5. Check all permutations finished correctly

for i in tqdm(range(n_null_cases)):
    ii = str(i).zfill(5)
    output_path  = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
    if not osp.exists(output_path):
        print("++ WARNING: %s is missing" % output_path)

# We will now compile all results into a single csv file for later exploration

df = pd.DataFrame(index=range(n_null_cases),columns=['Var. Exp. by Physio Regressors (NULL)'])
for i in tqdm(range(n_null_cases)):
    ii = str(i).zfill(5)
    model_path  = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
    model = sm.load(model_path)
    df.loc[i,'Var. Exp. by Physio Regressors (NULL)'] = model.rsquared
df=df.infer_objects()
df.index.name='Permutation'
df.to_csv('./cache/null_varexp_gs_physio.csv')

# ***
#
# # CODE TO DELETE
#
#
# # Check Results

df_null = pd.DataFrame(index=range(n_null_cases), columns=['Varexp. by Physio'])
for i in tqdm(range(n_null_cases)):
    ii = str(i).zfill(5)
    model_path  = osp.join(perm_dir,f'gs_phys_varex_{ii}.pkl')
    model = sm.load(model_path)
    df_null.loc[i,'Varexp. by Physio'] = model.rsquared
df_null = df_null.infer_objects()

df_null.hvplot.kde()

df = pd.DataFrame(index=selected_scans,columns=['Varexp. by Physio','kappa','rho','Mean Motion (enorm)','Max Motion (enorm)'])
for sbj,ses in tqdm(selected_scans):
    # Variance Explained by Regressors
    model_path  = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'pb03.{sbj}.r01.e02.volreg.scale.GSasis.PhysioModeling.pkl')
    model = sm.load(model_path)
    df.loc[(sbj,ses),'Varexp. by Physio'] = model.rsquared
    # Kappa and Rho
    kr_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'{sbj}_{ses}_GS_kappa_and_rho.txt')
    kr      = pd.read_csv(kr_path)
    df.loc[(sbj,ses),'kappa'] = kr.loc[0,'kappa']
    df.loc[(sbj,ses),'rho'] = kr.loc[0,'rho']
    # Motion
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D04_Preproc_fMRI_{ses}_NORDIC',f'motion_{sbj}_enorm.1D')
    aux_mot = np.loadtxt(mot_path)
    df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
    df.loc[(sbj,ses),'Max Motion (enorm)'] = aux_mot.max()
df =df.infer_objects()

df_null.hvplot.kde(label='Null Distribution') * df['Varexp. by Physio'].hvplot.kde(label='Real Data')

sample_sbj, sample_ses = df.sort_values(by='Varexp. by Physio', ascending=False).iloc[0].name

sample_gs_path = osp.join(PRCS_DATA_DIR,sample_sbj,f'D02_Preproc_fMRI_{sample_ses}',f'pb03.{sample_sbj}.r01.e02.volreg.scale.GSasis.1D')
sample_gs      = pd.DataFrame(np.loadtxt(sample_gs_path),columns=['orig'])
sample_gs.hvplot()

df.hvplot.scatter(x='Varexp. by Physio',y='kappa', aspect='square')

a = df_resp.loc[selected_scans]
a = a[['Mean','St.Dev.']]
a.columns = ['Resp Mean ipi','Resp St.Dev. ipi']
pd.concat([df,a],axis=1).hvplot.scatter(x='Varexp. by Physio',y='Resp St.Dev. ipi', aspect='square')

df_resp

a.loc[0,'kappa']

# ***
# # Computation of variance explained

osp.exists(osp.dirname("/data/SFIMJGC/ppp.j"))

# +
import statsmodels.api as sm

def variance_explained(df_target, df_predictors):
    # Ensure inputs are compatible
    y = df_target.iloc[:, 0]  # assuming only one column in target
    X = df_predictors

    # Add intercept term to predictors
    X = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X).fit()

    # R-squared: proportion of variance in y explained by X
    return model.rsquared, model


# -

def detrend_signal(y):
    nt    = len(y)
    trend = sm.add_constant(np.arange(nt))
    y_detrended = sm.OLS(y, trend).fit().resid
    return y_detrended
def variance_explained_after_trend(df_target, df_predictors):
    y = df_target.iloc[:, 0]
    X = df_predictors.copy()

    # Create linear trend regressor
    trend = sm.add_constant(np.arange(len(y)))

    # Regress out trend from y
    y_detrended = sm.OLS(y, trend).fit().resid

    # Regress out trend from each predictor
    X_detrended = X.apply(lambda col: sm.OLS(col, trend).fit().resid)

    # Add intercept to detrended predictors
    X_detrended = sm.add_constant(X_detrended)

    # Fit model on detrended data
    model = sm.OLS(y_detrended, X_detrended).fit()

    return model.rsquared, model


from utils.basics import detrend_signal

sbj_a,ses_a = selected_scans[10]
sbj_b,ses_b = selected_scans[100]

# Read and detrend global signal
gs_path = osp.join(PRCS_DATA_DIR,sbj_a,f'D02_Preproc_fMRI_{ses_a}',f'pb03.{sbj_a}.r01.e02.volreg.scale.GSasis.1D')
gs_df   = pd.DataFrame(np.loadtxt(gs_path))
gs_df.columns = ['orig']
gs_df['det'] = detrend_signal(gs_df['orig'].values.squeeze())

gs_df['det'].hvplot()

slibase_path = osp.join(PRCS_DATA_DIR,sbj_b,'D06_Physio',f'{sbj_b}_{ses_b}_task-rest_echo-2_slibase.1D')
slibase_obj  = Afni1D(slibase_path)
slibase_df   = pd.read_csv(slibase_path, comment='#', delimiter=' +', header=None, engine='python')
slibase_df.columns=slibase_obj.labels
slibase_df=slibase_df[3::].reset_index(drop=True)
slibase_det_df = slibase_df.copy()
for c in slibase_det_df.columns:
    slibase_det_df[c] = detrend_signal(slibase_df[c].values.squeeze())
phys_reg_list = np.unique([c.split('.',1)[1] for c in slibase_det_df.columns])

corrs_with_physio = pd.concat([gs_df,slibase_det_df],axis=1).corr().loc['det']
sel_physio_regs = []
for pr in phys_reg_list:
    aux = (corrs_with_physio.loc[[i for i in corrs_with_physio.index if pr in i]].abs().sort_values(ascending=False)).index[0]
    sel_physio_regs.append(aux)
sel_physio_regs

y = gs_df['det']
X = slibase_det_df[sel_physio_regs]
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()

model.rsquared

gs_df['det_pred'] = model.predict()

gs_df[['det','det_pred']].apply(zscore).hvplot()

slibase_df[['s000.card.s1','s001.card.s1','s002.card.s1']].hvplot(width=1000)

model.rsquared



slibase_path = osp.join(PRCS_DATA_DIR,sbj_b,'D06_Physio',f'{sbj_b}_{ses_b}_task-rest_echo-2_slibase.1D')
slibase_obj  = Afni1D(slibase_path)
slibase_df   = pd.read_csv(slibase_path, comment='#', delimiter=' +', header=None, engine='python')
slibase_df.columns=slibase_obj.labels
slibase_df=slibase_df[3::].reset_index(drop=True)

gs_df['det'] = detrend_signal(gs.values.squeeze())

gs_df['det'] = gs_det

gs['orig'].hvplot()

r2, model = variance_explained(gs_df, slibase_df[[c for c in slibase_df.columns if ('s000.card.c1') in c ]])
print(f"Variance explained (R²): {r2:.3f}")

r2, model = variance_explained_after_trend(gs_df, slibase_df[[c for c in slibase_df.columns if ('10.card.c1' in c) ]])
print(f"Variance explained (R²): {r2:.3f}")

slibase_df[[c for c in slibase_df.columns if '2.card.c1' in c ]].apply(zscore).hvplot(width=2000) * gs_df.apply(zscore).hvplot(width=2000).opts(line_width=5,line_color='k') 

from sklearn.metrics import explained_variance_score, r2_score

slibase_df['s000.card.c1'].values.squeeze().shape

r2_score(y_true=gs_df.values.squeeze(),y_pred=slibase_df['s000.card.c1'].values.squeeze())

np.corrcoef(gs_df.values.T,slibase_df['s000.card.c1'].values)

model.summary2()

slibase_df.columns





report_card_summary_df[('HR','Mean')]   = df[('peak ival over dset mean std','value_1')]
report_card_summary_df[('HR','St. Dev.')] = df[('peak ival over dset mean std','value_2')]

report_card_summary_df['peak num over dset'].hvplot.hist(bins=100, title='Distribution of number of cardiac peaks during a run', xlabel='Number of peaks',ylabel='# of Scans')

df['HR'].hvplot.scatter(x='Mean',y='St. Dev.', xlabel='Mean',ylabel='St. Dev.', title='Mean HR (b.p.m.)') * hv.VSpan(60,100)





report_card_keepers_df = read_gen_ss_review_table(report_card_keepers_path)

report_card_keepers_df.sort_values(by='peak num over dset').head(5)

df = pd.read_csv(report_card_summary_path, sep='\t', header=[0,1] )
original_columns  = [(a,b) for a,b in df.loc[[0,1]].values.T]
df.drop([0,1], inplace=True)
new_columns = []
df

# +
df = pd.read_csv(report_card_summary_path, sep='\t', header=[0,1] )
new_columns = []
current_lvl1 = None

for lvl1, lvl2 in df.columns:
    if "Unnamed" in str(lvl1):
        new_columns.append((current_lvl1, lvl2))
    else:
        current_lvl1 = lvl1
        new_columns.append((lvl1, lvl2))

df.columns = pd.MultiIndex.from_tuples(new_columns)
# -

df

original_columns

report_card_summary_df = read_gen_ss_review_table(report_card_summary_path)


def read_gen_ss_review_table(file_path):
    """
    Reads the output of AFNI command gen_ss_review_table and
    organizes the contents into a pandas dataframe with meaningful
    column names
    
    Input
    -----
    file_path: path to input file
    
    Returns
    -------
    df: dataframe with the information in the file
    """
    if not osp.exists(file_path):
        print('++ERROR [read_gen_ss_review_table]: input file does not exists')
        return None
    df = pd.read_csv(file_path, sep='\t', header=None)
    original_columns  = [(a,b) for a,b in df.loc[[0,1]].values.T]
    df.drop([0,1], inplace=True)
    new_columns = []
    for (a,b) in original_columns:
        if a=='infile':
            new_columns = new_columns + ['infile']
            continue
        if (a=='echo times'):
            new_columns = new_columns + ['e01','e02','e03']
            continue
        if (a is np.nan):
            continue
        if (a=='orig voxel counts'):
            new_columns = new_columns + ['Nx','Ny','Nz']
            continue
        if (a=='orig voxel resolution'):
            new_columns = new_columns + ['orig Dx','orig Dy','orig Dz']
            continue
        if (a=='final voxel resolution'):
            new_columns = new_columns + ['final Dx','final Dy','final Dz']
            continue
        if (a=='orig volume center'):
            new_columns = new_columns + ['orig volume center x','orig volume center y','orig volume center z']
            continue
        new_columns = new_columns + [a]
    df.columns = new_columns
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            df[c] = df[c]
    print("++ INFO [read_gen_ss_review_table]: Number of scans = %d | Number of metrics per scan = %d" % (df.shape))
    return df



# ## Check Quality of Physio

df = pd.DataFrame(index=scans_with_complete_physio,columns=['Resp Sampling Freq','Card Sampling Freq','Resp Peak IV Min','Resp Peak IV Max','Resp Peak IV Mean'])
for sbj,ses in scans_with_complete_physio:
    for ec in [2]:
        resp_review_path = osp.join(PRCS_DATA_DIR,sbj,'D06_Physio',f'{sbj}_{ses}_task-rest_echo-{ec}_resp_review.txt')
        resp_review = pd.read_csv(resp_review_path, sep=':', header=None, index_col=0, skipinitialspace=True)
        df.loc[(sbj,ses),'Resp Sampling Freq'] = np.float16(resp_review.loc['read_in sampling freq               '].values[0])
        df.loc[(sbj,ses),'Resp Peak IV Min']   = np.float16(resp_review.loc['peak ival min max                   '].values[0].split(' ')[0])
        df.loc[(sbj,ses),'Resp Peak IV Max']   = np.float16(resp_review.loc['peak ival min max                   '].values[0].split(' ')[1])
        df.loc[(sbj,ses),'Resp Peak IV Mean']  = np.float16(resp_review.loc['peak ival mean std                  '].values[0].split(' ')[0])
df = df.infer_objects()

df[['Resp Peak IV Min','Resp Peak IV Max','Resp Peak IV Mean']].hvplot(hover_cols=['Subject','Session'],width=1500)

resp_review


