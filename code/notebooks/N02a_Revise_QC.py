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

# # Description
#
# This notebook will use the outcome of AFNI QC reports to decide which scans we keep and which scans we will not include in sucessive analyses.

# The following ```get_ss_review_table``` command will generate two tables:
#
# * ```review_all_scans.txt```: contains summary QC values for all scans that completed afni_proc.py
# * ```review_keepers.txt```: contains information only for scans that pass our QC criteria.
#
# Our QC criteria is:
#
# 1. Only include scans from the GE scanner, becuase these seem to have slice timing information properly encoded (as opposed to the Siemens ones that do not): ```-report_outliers 'slice timing pattern' EQ simult```
# 2. Remove scans for which the EPI to Anat failed. Upon visual inspection we deem anat/EPI Dice <=.88 to reflect errors: ```-report_outliers 'anat/EPI mask Dice coef' LT 0.88```
# 3. Remove any scan that required an L/R Flip (None did, but nevertheless we leave the condition here): ```-report_outliers 'flip guess' EQ FLIP```
# 4. Remove any scan with dimensions different from those of the first scan (a few scans had voxel size slightly bigger (.001mm)). Most likely not an issue, but just to be on the safe side: ```-report_outliers 'orig voxel resolution' VARY```
# 5. For now, we do not remove data based on motion. The idea being, that we want to have both good and bad data to look at.
#    
# ```bash
# ml afni
# # cd /data/SFIMJGC_HCP7T/BCBL2024/prcs_data
# gen_ss_review_table.py -overwrite \
#                         -report_outliers 'anat/EPI mask Dice coef' LT 0.88 \
#                         -report_outliers 'slice timing pattern' EQ simult \
#                         -report_outliers 'flip guess' EQ FLIP \
#                         -report_outliers 'orig voxel counts' VARY \
#                         -report_outliers 'orig voxel resolution' VARY \
#                         -report_outliers 'TR' VARY \
#                         -report_outliers 'num TRs per run' VARY \
#                         -report_outliers 'echo times' VARY \
#                         -report_outliers 'average motion (per TR)' SHOW \
#                         -report_outliers 'max censored displacement' SHOW \
#                         -report_outliers 'max motion displacement'   SHOW \
#                         -report_outliers 'num TRs per run (applied)' SHOW \
#                         -report_outliers 'TSNR average' SHOW \
#                         -report_outliers 'global correlation (GCOR)' SHOW \
#                         -report_outliers 'anat/templ mask Dice coef' SHOW \
#                         -report_outliers 'censor fraction' SHOW \
#                         -write_table review_all_scans.txt \
#                         -write_outliers review_keepers.txt -show_keepers \
#                         -infiles ./sub-*/D03_Preproc_ses-?_NORDIC-off/out.ss_review.*
# ```
#
# Not using this for now, as we want to have all possible motion here
#
#                        -report_outliers 'censor fraction' GT 0.1 \

# Data from this dataset was acquired in two different scanners:
#
# * GE scanner --> Seems to have slice timing information available
# * Siemens scanner --> Slice timing information is missing from the headers.

import pandas as pd
import numpy as np
import hvplot.pandas
import os.path as osp
from utils.basics import PRJ_DIR

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)
from utils.basics import read_gen_ss_review_table

report_summary_path  = osp.join(PRJ_DIR,'prcs_data','review_all_scans.txt')
report_keepers_path  = osp.join(PRJ_DIR,'prcs_data','review_keepers.txt')

# # 1. Load the Full Report
#
# Sometimes we will want to compare what we keep to the full sample. It is for these purposes that we load the full report

report_summary_df = read_gen_ss_review_table(report_summary_path)

# As we shall see in later notebooks, there will be three subjects for whom one scan passed criteria and another one did not. The next three cells show that the cause for that is the EPI/anat overlap being below 0.88

report_summary_df.set_index('subject ID').loc['sub-12']

report_summary_df.set_index('subject ID').loc['sub-142']

report_summary_df.set_index('subject ID').loc['sub-53']

# ***
#
# # 2. Load the list of scans that passed QC criteria

report_keepers_df = read_gen_ss_review_table(report_keepers_path)
report_keepers_df.head(5)

good_scan_ids = list(report_keepers_df['infile'])
print("++ Number of selected scans: %d scans" % len(good_scan_ids))

print("++ Number of subjects: %d subjects" % np.unique([row['infile'].split('/')[1] for r,row in report_keepers_df.iterrows()]).shape[0])

report_keepers_df.head(1)

# # 3. Check motion

a = report_keepers_df.hvplot.kde('average motion (per TR)',label='Passed QC', alpha=.5, title='Average Motion (per TR)') * report_summary_df.hvplot.kde('average motion (per TR)', label='All scans', alpha=.5)
b = report_keepers_df.hvplot.hist('average motion (per TR)', bins=np.linspace(0,0.4,50), label='Passed QC', normed=True, alpha=.5, title='Average Motion (per TR)') * \
report_summary_df['average motion (per TR)'].hvplot.hist('average motion (per TR)', bins=np.linspace(0,0.4,50), label='All scans', alpha=.5, normed=True)
a+b

a = report_keepers_df.hvplot.kde('max motion displacement',label='Passed QC', alpha=.5, title='Maximum Motion Displacement') * report_summary_df.hvplot.kde('max motion displacement', label='All scans', alpha=.5)
b = report_keepers_df.hvplot.hist('max motion displacement', bins=np.linspace(0,6,50), label='keepers', normed=True, alpha=.5, title='Maximum Motion Displacement') * \
report_summary_df['max motion displacement'].hvplot.hist('max motion displacement', bins=np.linspace(0,6,50), label='all', alpha=.5, normed=True)
a+b

report_keepers_df.columns

worse_subject = report_keepers_df.sort_values(by='num TRs per run (applied)').reset_index().iloc[0]['infile']
best_subject = report_keepers_df.sort_values(by='num TRs per run (applied)').reset_index().iloc[-1]['infile']
print('++ INFO: Best subject: %s' % best_subject)
print('++ INFO: Worse subject: %s' % worse_subject)

# Most likely we will use 'max censored displacement' as the estimate of motion when doing the Powers et al. business. As according to AFNI Discourse (https://discuss.afni.nimh.nih.gov/t/max-motion-displacement-question/2754/3)... "The censored displacement just considers time points that were not removed, those that are still in the time series after censoring. It is more useful, since those are the time points that will be considered in the regression."

# # 4. Check TSNR

report_keepers_df.hvplot.hist('TSNR average', title='Distribution of TSNR in final sample')

# # 5. Check Alignment Quality

report_keepers_df.hvplot.hist('anat/EPI mask Dice coef', title='Quality of EPI - Anat Overlap', normed=True, alpha=0.5, bins=np.linspace(.7,1,50)) * \
report_summary_df['anat/EPI mask Dice coef'].hvplot.hist('anat/EPI mask Dice coef', title='Quality of EPI - Anat Overlap', normed=True, alpha=0.5, bins=np.linspace(.7,1,50))

report_keepers_df.hvplot.hist('anat/templ mask Dice coef', title='Quality of Transformation to MNI')

# # 6. Check other metrics

report_keepers_df.hvplot.hist('global correlation (GCOR)', title='global correlation (GCOR)')

report_keepers_df[['e01','e02','e03']].hvplot.box()

# # 7. Write final list of scans to disk

sbj_idx = [i.split('/')[1] for i in report_keepers_df['infile'].values]
ses_idx = [i.split('/')[2].split('_')[-1] for i in report_keepers_df['infile'].values]
report_keepers_df.index = pd.MultiIndex.from_arrays([sbj_idx,ses_idx],names=['Subject','Session'])

report_keepers_df.to_csv('../../../resources/good_scans.txt')
