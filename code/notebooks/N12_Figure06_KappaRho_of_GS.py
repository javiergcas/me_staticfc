#!/usr/bin/env python
# coding: utf-8

# # 1. Setup and Data Preparation
# 
# ## 1.1. Objective
# 
# Build Figure 06 by combining two complementary views of global-signal behavior across scans:
# - Panel (a): relationship between GS kappa and GS rho, with motion encoded by marker size/color.
# - Panel (b): distribution of adjusted R^2 values for physiological variance explained in GS, with a non-parametric significance threshold.

# In[1]:


import os.path as osp
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler 
from utils.basics import get_dataset_index
from utils.basics import PRCS_DATA_DIR
from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import panel as pn
hv.extension('bokeh')
pn.extension('mathjax')


# ## 1.2. Imports and Configuration
# 
# The next cells import analysis/plotting libraries, select the target dataset (`evaluation`), and build the subject-session index used to align all scan-level tables.

# In[2]:


DATASET='evaluation'


# In[3]:


ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())


# ## 1.3. Load Input Tables and Motion Metrics
# 
# The next cells load GS kappa/rho estimates, real adjusted R^2 estimates, and the null adjusted R^2 distribution. They also compute scan-level motion summaries (mean/max ENORM) and scaled marker-size columns for panel (a).
# 

# In[7]:


gs_kappa_rho_path = f'./summary_files/{DATASET}_gs_kappa_rho.csv'
gs_kappa_rho_df = pd.read_csv(gs_kappa_rho_path, index_col=[0,1])
print('++ INFO: GS Kappa and Rho estimates loaded from %s' % gs_kappa_rho_path)
print("++ INFO: The shape of kappa_rho_df is %s" % str(gs_kappa_rho_df.shape))
gs_kappa_rho_df.head(3)


# # 1.4. Load Variance Explained in the Global Signal by Physiological Regressors

# In[8]:


real_varex_path = f'./summary_files/{DATASET}_varexp_gs_physio.real_data.csv'
real_varex_df   = pd.read_csv(real_varex_path, index_col=[0,1])
print('++ INFO: Estimates of variance explained from the GS by physio regressors loaded from %s' % real_varex_path)
print("++ INFO: The shape of real_varex_df is %s" % str(real_varex_df.shape))
real_varex_df.head(3)


# ## 1.5. Load a null distribution for the variance explained in the GS

# In[9]:


null_varex_path = f'./summary_files/{DATASET}_varexp_gs_physio.null_distribution.csv'
null_varex_df   = pd.read_csv(null_varex_path, index_col=[0])
print('++ INFO: Null distribution for estimates of variance explained from the GS by physio regressors loaded from %s' % null_varex_path)
print("++ INFO: The shape of null_varex_df is %s" % str(null_varex_df.shape))
null_varex_df.head(3)


# ## 1.6. Load Head Motion Estimates

# In[10]:


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
print (motion_df.shape)


# ***
# # 2. Generate Panel (a): Scan-Level Kappa vs Rho of GS
# 
# ## 2.1. Build plotting table
# 
# Concatenate GS kappa/rho values with motion metrics so each row corresponds to one scan with all variables needed for visualization.

# In[11]:


df = pd.concat([gs_kappa_rho_df, motion_df], axis=1)
df.head(3)


# 
# ## 2.2. Create and export static panel
# 
# Construct a Seaborn JointGrid scatter of `kappa (GS)` vs `rho (GS)`, add diagonal reference regions, marginal distributions, and a motion colorbar, then save panel (a) as `./figures/pBOLD_Figure06_a.png`.

# In[12]:


# Set seaborn theme
sns.set_theme(style="white", context="notebook")

# Compute quantile limits for color normalization
vmin = df["Max. Motion (enorm)"].quantile(0.05)
vmax = df["Max. Motion (enorm)"].quantile(0.99)

# Create JointGrid
g = sns.JointGrid(data=df, x="kappa (GS)", y="rho (GS)", space=0, height=8)

# Define axes limits for the diagonal line
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

#sizes = df["Max. Motion (dot size)"]
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

# Save to disk
plt.tight_layout()
plt.close()
# After all plotting code (and before/without plt.show())
out_png = "./figures/pBOLD_Figure06_a.png"

g.figure.savefig( out_png, dpi=300, bbox_inches="tight",   # critical: captures cbar outside default figure bounds
    pad_inches=0.05)

print(f"Saved: {out_png}")


# ***
# 
# # 3. Generate Panel (b): Distribution of Adjusted R^2
# 
# ## 3.1. Estimate significance threshold from null data
# 
# Compute the non-parametric `p=0.05` cutoff as the 95th percentile of the null adjusted R^2 distribution.

# In[14]:


non_parametric_p005 = null_varex_df.quantile(0.95).values[0]
print('++ INFO: Non-parametric value for p=0.05 --> %.2f' % non_parametric_p005)


# 
# ## 3.2. Create annotated histogram/KDE view
# 
# Generate histogram and KDE curves for real adjusted R^2 values, shade the statistically significant range (`R^2 >= threshold`), and add the horizontal arrow annotation label.
# 
# The hook function in the next cell is used for the horizontal annotation.
# 

# In[15]:


x0, x1 = non_parametric_p005, 1.0
y_ann = 4.5
def add_span_label(plot, element):
    ax = plot.handles["axis"]
    xm = 0.5 * (x0 + x1)
    gap = 0.4 * (x1 - x0)  # small gap around text

    # Left and right arrows (so arrows are not part of the text string)
    ax.annotate(
        "", xy=(x0, y_ann), xytext=(xm - gap, y_ann),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        annotation_clip=False
    )
    ax.annotate(
        "", xy=(x1, y_ann), xytext=(xm + gap, y_ann),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        annotation_clip=False
    )

    ax.text(
        xm, y_ann, r"Statistically Significant Adjusted $R^{2}$ values",
        ha="center", va="center", fontsize=16, color="black",
        zorder=10
    )


# In[16]:


hv.extension('matplotlib')
panel_b = (hv.VSpan(non_parametric_p005,1.0).opts(facecolor='lightgray', color='gray') * \
real_varex_df.hvplot.hist(bins=20, xlim=(0,1),                     ylabel=' % Scans', xlabel=r"$Adjusted R^{2}$",normed=True, label='Real Data', legend=False, fontscale=1.5, facecolor='gray') * \
real_varex_df.hvplot.kde(          xlim=(0,1),                     ylabel=' % Scans',              label='Real Data', legend=False, facecolor='gray'))
panel_b = panel_b.opts(hooks=[add_span_label])


# ***
# # 4. Assemble Figure 06 Layout
# 
# Combine panel (a) PNG and panel (b) HoloViews output into one Panel column, add `(a)`/`(b)` labels, and save the final figure as `./figures/pBOLD_Figure06.html`.
# 

# In[17]:


pn.Column( pn.Row(pn.pane.Markdown('## (a)'), width=10),
                  pn.pane.PNG('./figures/pBOLD_Figure06_a.png', width=700),
           pn.Row(pn.pane.Markdown('## (b)'), width=10),
                  pn.panel(panel_b, width=700)         
        ).save('./figures/pBOLD_Figure06.html')


# ***
# # 5. Final Output Preview
# 
# Static preview of the final figure:
# 
# ![Figure 06](figures/pBOLD_Figure06.png)
# 

# 
