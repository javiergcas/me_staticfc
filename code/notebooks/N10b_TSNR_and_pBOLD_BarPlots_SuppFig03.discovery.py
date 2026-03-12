#!/usr/bin/env python
# coding: utf-8

# # Description: Create Panels a and b for Suplementary Figure 03
# 
# This notebook will load all TSNR and pBOLD estimates for the ```discovery``` dataset and create summary bar plots with statistics

# In[1]:


import os.path as osp
import xarray as xr
import pandas as pd
import pickle 
from utils.basics import CODE_DIR
from utils.dashboard import get_static_report


# Select dataset and FC metric

# In[2]:


DATASET='discovery'
FC_METRIC='cov'


# ***
# # 1. Load pBOLD
# 
# Create path to the file with all pBOLD estimates (previously generated in ```N08_Gather_Results.ipynb```)

# In[3]:


pBOLD_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_pBOLD.nc')
print('++ Reading pBOLD from %s' % pBOLD_path)


# Load the xr DataArray with the pBOLD estimates

# In[4]:


pBOLD = xr.open_dataarray(pBOLD_path)


# Extract only the scan-level pBOLD values and place them into a dataframe structured ready for plotting purposes

# In[5]:


pBOLD = pBOLD.sel(fc_metric=FC_METRIC,ee_vs_ee='scan')
pBOLD = pBOLD.to_dataframe(name='pBOLD').reset_index().drop(['qc_metric','fc_metric','ee_vs_ee'],axis=1)
pBOLD.columns=['Subject','Session','Pre-processing','m-NORDIC','pBOLD']
pBOLD = pBOLD
pBOLD.head(3)


# ***
# # 2. Load TSNR
# 
# Create path to the file with all TSNR estimates (previously generated in ```N08_Gather_Results.ipynb```)

# In[6]:


TSNR_path = osp.join(CODE_DIR,'notebooks','summary_files',f'{DATASET}_TSNR.pkl')
print('++ TSNR will read from %s' % TSNR_path)


# Load into memory

# In[7]:


with open(TSNR_path, 'rb') as f:
    TSNR = pickle.load(f)


# Rename NORDIC to m-NORDIC so labels agree with the way we refer to magnitude-only NORDIC throughout the manuscript

# In[8]:


TSNR[('cov','TSNR (Full Brain)')].columns=['Subject','Session','Pre-processing','m-NORDIC','TSNR (Full Brain)']
TSNR[('corr','TSNR (Full Brain)')].columns=['Subject','Session','Pre-processing','m-NORDIC','TSNR (Full Brain)']


# ***
# # 3. Combine into single dictionary
# 
# We now combine all loaded data into a single dictionary
# 

# In[9]:


QC_metrics = TSNR.copy()
QC_metrics[FC_METRIC,'pBOLD'] = pBOLD


# ***
# # 4. Create Suppl Figure 03 | Panel a
# 
# In Panel a we show Full Brain TSNE across all pipelines. We show statistics based on the Mann-Whitney test. Finally, we include both sessions (1 and 2) by setting ```SES='all'```

# In[26]:


QC_METRIC = 'TSNR (Full Brain)'
SEL_PIPELINES = ['ALL_Basic','ALL_GS','ALL_Tedana-fastica']
STAT_TEST = 'Mann-Whitney'
SES = 'constant_gated'


# We create panel a, bar plots for TSNR, using the get_static_report function

# In[27]:


if SES == 'all':
    data_to_show = QC_metrics[FC_METRIC,QC_METRIC].set_index('Pre-processing').loc[SEL_PIPELINES].reset_index()
else:
    idx = pd.IndexSlice
    data_to_show = QC_metrics[FC_METRIC,QC_METRIC].set_index(['Pre-processing','Session']).loc[idx[SEL_PIPELINES, SES], :].reset_index()
panel_a = get_static_report(data_to_show,
                  FC_METRIC,
                  QC_METRIC,  
                  x='Pre-processing',
                  hue='m-NORDIC',
                  stat_test=STAT_TEST, 
                  show_stats=True, 
                  show_points=True,
                  stat_annot_type='star', 
                  remove_outliers_from_swarm=True, 
                  legend_location='lower left', 
                  session=SES, dot_size=1)
panel_a.suptitle("(a)", x=0.01, ha="left", fontsize=20);
panel_a


# In[28]:


panel_a.tight_layout()
panel_a.savefig('./figures/pBOLD_SuppFig03_a.png', bbox_inches="tight", pad_inches=0)


# ***
# # 5. Create Suppl. Fig 03 | Panel b
# 
# Similar to section 4, but this time we work with pBOLD instead of TSNR

# In[29]:


QC_METRIC     = 'pBOLD'
SEL_PIPELINES = ['ALL_Basic','ALL_GS','ALL_Tedana-fastica']
STAT_TEST     = 'Mann-Whitney'
SES           = 'constant_gated'


# In[30]:


if SES == 'all':
    data_to_show = QC_metrics[FC_METRIC,QC_METRIC].set_index('Pre-processing').loc[SEL_PIPELINES].reset_index()
else:
    idx = pd.IndexSlice
    data_to_show = QC_metrics[FC_METRIC,QC_METRIC].set_index(['Pre-processing','Session']).loc[idx[SEL_PIPELINES, SES], :].reset_index()
panel_b = get_static_report(data_to_show,
                  FC_METRIC,
                  QC_METRIC,  
                  x='Pre-processing',
                  hue='m-NORDIC',
                  stat_test=STAT_TEST, 
                  show_stats=True, 
                  show_points=True,
                  stat_annot_type='star', 
                  remove_outliers_from_swarm=True, 
                  legend_location='lower left', 
                  session=SES, dot_size=1)
panel_b.suptitle("(b)", x=0.01, ha="left", fontsize=20);
panel_b


# In[31]:


panel_b.tight_layout()
panel_b.savefig('./figures/pBOLD_SuppFig03_b.png', bbox_inches="tight", pad_inches=0)

