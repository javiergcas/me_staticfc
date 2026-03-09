#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path as osp
import pandas as pd
import numpy as np
import os
import copy
from tqdm import tqdm
import datetime
from utils.basics import PRJ_DIR, PRCS_DATA_DIR,CODE_DIR,NUM_DISCARDED_VOLUMES
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


# In[2]:


# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)


# In[3]:


DATASET='evaluation'
CENSOR_MODE = 'ALL'


# In[4]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# In[5]:


fMRI_num_discarded_volumes = NUM_DISCARDED_VOLUMES[DATASET]


# ***
# # 1. Load data
# 
# ### 1.1. Kappa and Rho for Global Signal

# In[6]:


kappa_rho_df = pd.read_csv(f'./cache/{DATASET}_gs_kappa_rho.{CENSOR_MODE}.csv', index_col=[0,1])
print("++ INFO: The shape of kappa_rho_df is %s" % str(kappa_rho_df.shape))
kappa_rho_df.head(2)


# ### 1.2. Load Variance Explained by Physiological Regressors
# 
# > **NOTE:** This dataframe contains less entries than the one above because good physio was not available for all scans.

# In[7]:


real_varex_df = pd.read_csv('./cache/real_varexp_gs_physio.csv', index_col=[0,1])
print("++ INFO: The shape of real_varex_df is %s" % str(real_varex_df.shape))
real_varex_df.head(2)


# We also load the null distribution of variance explained in GS by physio regressors

# In[8]:


null_varex_df = pd.read_csv('./cache/null_varexp_gs_physio.csv', index_col=[0])
print("++ INFO: The shape of null_varex_df is %s" % str(null_varex_df.shape))
null_varex_df.head(2)


# ### 1.3. Load Head Motion estimates

# In[9]:


mms = MinMaxScaler(feature_range=(2, 100))
motion_df = pd.DataFrame(index=ds_index, columns = ['Mean Motion (enorm)','Max. Motion (enorm)'])
for sbj,ses in tqdm(ds_index):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'motion_{sbj}_enorm.1D')
    if osp.exists(mot_path):
        aux_mot = np.loadtxt(mot_path)
        motion_df.loc[(sbj,ses),'Mean Motion (enorm)'] = aux_mot.mean()
        motion_df.loc[(sbj,ses),'Max. Motion (enorm)'] = aux_mot.max()
motion_df = motion_df.infer_objects()
motion_df['Mean Motion (dot size)'] = mms.fit_transform(motion_df['Mean Motion (enorm)'].values.reshape(-1,1))
motion_df['Max. Motion (dot size)'] = mms.fit_transform(motion_df['Max. Motion (enorm)'].values.reshape(-1,1))


# ### 1.4. Load TSNR and pBOLD for all scans

# In[10]:


import pickle
with open(f'./cache/{DATASET}_QC_metrics_{CENSOR_MODE}.pkl', 'rb') as f:
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

# In[11]:


df = pd.concat([kappa_rho_df, motion_df], axis=1)
df.head(3)


# In[12]:


kappa_vs_rho_byType = df.hvplot.scatter(x='kappa (GS)',y='rho (GS)', aspect='square', hover_cols=['Subject','Session'], c='kappa_rho_color',s='Mean Motion (dot size)', title='Global Signal Kappa vs. Rho') *hv.Slope(1,0).opts(line_width=0.5,line_dash='dashed', line_color='black')


# In[13]:


cbar_min = df['Max. Motion (enorm)'].quantile(0.05)
cbar_max = df['Max. Motion (enorm)'].quantile(0.99)
kappa_vs_rho_byMotion = df.hvplot.scatter(x='kappa (GS)',y='rho (GS)', aspect='square', hover_cols=['Subject','Session'], c='Max. Motion (enorm)', title='Global Signal Kappa vs. Rho', cmap='cividis').opts(clim=(cbar_min,cbar_max)) *hv.Slope(1,0).opts(line_width=0.5,line_dash='dashed', line_color='black')


# In[14]:


kappa_vs_rho_byType + kappa_vs_rho_byMotion


# In[15]:


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
g = sns.JointGrid(data=df, x="kappa (GS)", y="rho (GS)", space=0, height=8)
#g = sns.JointGrid(data=df, y="kappa (GS)", x="rho (GS)", space=0, height=8)

# Define axes limits for the diagonal line
#xlim = g.ax_joint.get_xlim()
#ylim = g.ax_joint.get_ylim()
min_val = 0
max_val = 125

# Create diagonal line (y = x)
line_x = np.linspace(min_val, max_val, 500)
line_y = line_x

# Fill above (light red) and below (light green)
g.ax_joint.fill_between(line_x, line_y, max_val, color="#FFCCCC", alpha=0.4)  # Light red above
g.ax_joint.fill_between(line_x, min_val, line_y, color="#CCFFCC", alpha=0.4)  # Light green below

# Draw the diagonal line
g.ax_joint.plot(line_x, line_y, color="black", linestyle="--", linewidth=1)

# Scatter plot with color and size
#df = df[["pBOLD", "TSNR (Full Brain)", "Max. Motion (dot size)", "Max. Motion (enorm)"]].dropna()

#sizes = df["Max. Motion (dot size)"]
#scaled_sizes = 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min()) + 10  # Scale between 10 and 210
sc = g.ax_joint.scatter(
    data=df,
    x="kappa (GS)",
    y="rho (GS)",
    s=df['Max. Motion (dot size)'],
    c=df["Max. Motion (enorm)"],
    cmap="cividis",
    vmin=vmin, vmax=vmax,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.3,
)

# Create new axes above the top marginal histogram
pos_joint = g.ax_joint.get_position()
pos_marg_x = g.ax_marg_x.get_position()
cbar_ax = g.fig.add_axes([pos_joint.x0, pos_marg_x.y1 + 0.05, pos_joint.width, 0.02])  # [left, bottom, width, height]

# Add horizontal colorbar
cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')

# Move label and ticks to the top
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.tick_top()
cbar.set_label("Max. Motion (mm)", rotation=0, labelpad=5)

# Top and right marginal plots
sns.histplot(x=df["kappa (GS)"], ax=g.ax_marg_x, kde=True, color="green", edgecolor="black", bins=50)
g.ax_marg_x.axvline(
    df["kappa (GS)"].median(), color="black", linewidth=3, linestyle='--', label='Median'
)
sns.histplot(y=df["rho (GS)"], ax=g.ax_marg_y, kde=True, color="red", edgecolor="black", bins=50)
g.ax_marg_y.axhline(
    df["rho (GS)"].median(), color='black', linewidth=3, linestyle='--', label='Median'
)
# Axis labels
g.set_axis_labels("kappa (GS)", "rho (GS)")

# Improve layout
g.ax_joint.set_xlim(12,125)
g.ax_joint.set_ylim(12,125)
plt.show()


# ***
# # 3. Explore Variance Explained by Physiological Regressors
# 
# ### 3.1. How much variance do physio regressors explained in our sample?

# In[16]:


non_parametric_p005 = null_varex_df.quantile(0.95).values[0]
print(non_parametric_p005)


# In[19]:


#dashboard.stop()
graph = (hv.VSpan(non_parametric_p005,1.0).opts(fill_color='lightgray', line_color='white') * \
real_varex_df.hvplot.hist(bins=20, xlim=(0,1),                     ylabel=' % Scans', xlabel="$$Adj. R2$$",normed=True, label='Real Data', legend=False, fontscale=1.5, color='gray') * \
real_varex_df.hvplot.kde(          xlim=(0,1),                     ylabel=' % Scans',              label='Real Data', legend=False, color='gray'))
dashboard = pn.Row(graph).show()


# In[20]:


from numpy import arange
from bokeh.plotting import figure, show

x = arange(1, 4.5, 0.25)
y = 1 / x
plot = figure(height=200)
plot.circle(x, y, fill_color="blue", size=5)
plot.line(x, y, color="darkgrey")
plot.xaxis.axis_label = "Resistance"
plot.xaxis.ticker = [1, 2, 3, 4]
plot.yaxis.axis_label = "Current at 1 V"
plot.xaxis.major_label_overrides = {
    1: r"$$1\\ \\Omega$$",
    2: r"$$2\\ \\Omega$$",
    3: r"$$3\\ \\Omega$$",
    4: r"$$4\\ \\Omega$$",
}
show(plot)


# In[21]:


num_scans_with_physio_not_significantly_explaining_any_variance = (real_varex_df>=non_parametric_p005).sum().values[0]
pc_scans_with_physio_not_significantly_explaining_any_variance = 100 * num_scans_with_physio_not_significantly_explaining_any_variance / real_varex_df.shape[0]
print("++ INFO: Number and [percentage] of scans for which physio regressors explain a signficiant amount of variance: %d [ %.2f%% ]" % (num_scans_with_physio_not_significantly_explaining_any_variance,pc_scans_with_physio_not_significantly_explaining_any_variance))


# In[22]:


real_varex_df.mean()


# In[23]:


hv.VSpan(non_parametric_p005,1.0).opts(fill_color='gray') * \
real_varex_df.hvplot.hist(bins=20, xlim=(0,1),                     ylabel=' % Scans', xlabel=r'$$adjusted R^{2}$$',normed=True, label='Real Data', legend=False, fontscale=1.5) * \
real_varex_df.hvplot.kde(          xlim=(0,1),                     ylabel=' % Scans',              label='Real Data', legend=False) * \
hv.Text(0.5,7,'p > 0.05') + \
null_varex_df.hvplot.hist(bins=20, xlim=(0,1), title='Variance Explained by Physio Regressors in the Global Signal (NULL Distribution)', ylabel=' % Scans', normed=True, label='Null Distribution', color='gray') * \
null_varex_df.hvplot.kde(          xlim=(0,1), title='Variance Explained by Physio Regressors in the Global Signal',                     ylabel=' % Scans',              label='Null Distribution', color='gray')


# In[23]:


(real_varex_df > 0.5).sum()


# In[41]:


v = float(real_varex_df.loc['sub-158','ses-1'].values[0])
real_varex_df.sort_values(by='Var. Exp. by Physio Regressors').reset_index(drop=True).hvplot() * hv.HLine(v)


# In[39]:


float(real_varex_df.loc['sub-158','ses-1'].values[0])


# In[ ]:




