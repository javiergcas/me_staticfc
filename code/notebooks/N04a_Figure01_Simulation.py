#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# This notebook contains the code used to generate Figure 1, which exemplifies how covariance-based FC behaves across echoes in the two extreme signal regimes

# In[1]:


from utils.basics import PRCS_DATA_DIR, TES_MSEC
import os.path as osp
import numpy as np
import pandas as pd
from nilearn.connectome import sym_matrix_to_vec
import hvplot.pandas
import holoviews as hv
import panel as pn
import holoviews as hv
hv.extension("matplotlib")
pn.extension()


# ***
# 
# # 1. Non-BOLD Behavior
# 
# The code below generates Figure 1.a where we show both a simulation and real case of how data that is non-BOLD dominated behaves.
# 
# First, we load real non-constant TR data (i.e., one random scan from the discovery dataset), with minimal pre-processing. This an example of non-BOLD dominated data

# In[2]:


sbj        = 'MGSBJ03'       # Subject ID 
ses        = 'cardiac_gated' # Session ID
scenario   = 'ALL_Basic'     # Basic denoising
ATLAS_NAME = 'Power264-discovery' # Atlas ID used for the discovery dataset (only ROIs in the FOV for this dataset)
tes        = ['e01','e02','e03']
te_labels  = {'e01':'TE1','e02':'TE2','e03':'TE3'}


# ## 1.1. Load timeseries for all available TEs:

# In[3]:


roi_ts = {}
for te in tes:
    roi_ts_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.{te}.volreg.spc.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts[te] = np.loadtxt(roi_ts_path)


# ## 1.2. Compute the covariance matrix for all pairs of TEs:

# In[4]:


FC_c    = {}
for te_x in tes:
    for te_y in tes:
        aux_ts_x = roi_ts[te_x]
        aux_ts_y = roi_ts[te_y]
        aux_fc_c = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_y.shape[1]:]
        FC_c[(te_x,te_y)] = sym_matrix_to_vec(aux_fc_c,discard_diagonal=True)


# ## 1.3 Generate Simulated Scatter Plot
# 
# To generate a "cartoonish" scatter plot that exemplifies the non-BOLD behavior--namely points sitting over the identity line--we do the following:
# 
# 1) ```x_data```: We load real FC_c for a pair of echoes (```(e01,e02)```)
# 2) ```y_data```: Instead of loading FC_c for a second pair of echoes, we simulate the FC_c for that second pair of echoes by adding a bit of noise to the one loaded in 1.

# In[5]:


x_data            = FC_c[('e01','e02')]
x_data_label      = 'C(TEi,TEj)'
x_data_label_plot =  r"$C(TE_{i},TE_{j})$"

y_data            = x_data + 0.1 * (np.random.normal(0,.25,25425))
y_data_label      = 'C(TEk,TEl)'
y_data_label_plot =  r"$C(TE_{k},TE_{l})$"


# Next, we load the two sets of FC_c into a single pandas dataframe for plotting purposes

# In[6]:


non_bold_sim_df = pd.DataFrame([x_data,y_data], 
                  index=[x_data_label, y_data_label]).T


# In[7]:


non_bold_sim_scatter = non_bold_sim_df.hvplot.scatter(x=x_data_label,y=y_data_label, 
                                   aspect='square',
                                   color='black', 
                                   datashade=True, 
                                   xlabel=x_data_label_plot, 
                                   ylabel=y_data_label_plot, 
                                   xlim=(-.1,1.5), 
                                   ylim=(-.1,1.5),
                                   title='(a) Simulation',
                                   width=250).opts(fontscale=1.0)    


# ## 1.4. Generate Real Data Scatter Plot
# 
# In this case we load real data for both the x and y axis of the plot

# In[8]:


x_data            = FC_c[('e01','e02')]
x_data_label      = f"C({te_labels['e01']},{te_labels['e02']})"
x_data_label_plot =  r"$C( TE_{1},TE_{2})$"

y_data            = FC_c[('e01','e03')]
y_data_label      = f"C({te_labels['e01']},{te_labels['e03']})"
y_data_label_plot =  r"$C( TE_{1},TE_{3})$"


# In[9]:


non_bold_real_df = pd.DataFrame([x_data,y_data], 
                                index=[x_data_label, y_data_label]).T


# In[10]:


non_bold_real_scatter = (
    non_bold_real_df.hvplot.scatter(
        x=x_data_label,
        y=y_data_label,
        aspect="square",
        color="black",
        datashade=True,  # if this errors on mpl, switch to rasterize=True
        xlabel=x_data_label_plot,
        ylabel=y_data_label_plot,
        xlim=(-0.1, 1.5),
        ylim=(-0.1, 1.5),
        title="(b) Real Data",
        width=250
    )
    .opts(fontscale=1.0)
)


# # 1.5 Generate additional graphic elements for Panels (a) and (b)
# First, we create the reference axes for (0,0)

# In[11]:


zero_marker = (
    hv.VLine(0).opts(color="gray", linestyle="dashed", linewidth=0.5)
    * hv.HLine(0).opts(color="gray", linestyle="dashed", linewidth=0.5)
)


# Second, we create a red dashed line that has slope = 0 and intersect = 1. We also create the text annotation that accompanies that line.
# 
# This is the expected behavior for the non-BOLD dominant regime

# In[12]:


non_BOLD_line = hv.Slope(1, 0).opts(color="red", linestyle="dashed", linewidth=2)

def add_nonbold_label(plot, element):
    ax = plot.handles["axis"]
    ax.text(
        1.0, 0.7, "y = 1 \u22C5 x + 0",
        color="red",
        fontsize=12,
        rotation=45,
        ha="left",
        va="bottom",
    )


# Third, we compose the individual panels by adding the non_BOLD_line and the zero marker to each of the scatter plot

# In[13]:


non_bold_sim_panel  = (non_bold_sim_scatter  * zero_marker * non_BOLD_line).opts(hooks=[add_nonbold_label])
non_bold_real_panel = (non_bold_real_scatter * zero_marker * non_BOLD_line).opts(hooks=[add_nonbold_label])


# Forth, we generate the top banner signaling these are results for the non-BOLD dominated scenario

# In[14]:


non_bold_banner = pn.pane.HTML(
    """
    <div style="
        width:100%;
        background:#000;
        color:#fff;
        font-weight:700;
        text-align:center;
        padding:10px 12px;
        box-sizing:border-box;
        font-size:24px;
    ">
      Non-BOLD Dominated Data
    </div>
    """,
    sizing_mode="scale_width",
)


# ***
# 
# # 2. BOLD-Dominated Behavior
# 
# In this case we load data that was acquired with a regular TR and has been aggresively denoised with tedana. 
# 
# This is an example of BOLD dominated data

# In[15]:


sbj='MGSBJ03'
ses='constant_gated'
scenario   = 'ALL_Tedana-fastica'


# ## 2.1. Load timeseries for all available TEs:

# In[16]:


roi_ts = {}
for te in tes:
    roi_ts_path = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.{te}.volreg.spc.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts[te] = np.loadtxt(roi_ts_path)


# ## 2.2. Compute the covariance matrix for all pairs of TEs:

# In[17]:


FC_c    = {}
for te_x in tes:
    for te_y in tes:
        aux_ts_x = roi_ts[te_x]
        aux_ts_y = roi_ts[te_y]
        aux_fc_c = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_y.shape[1]:]
        FC_c[(te_x,te_y)] = sym_matrix_to_vec(aux_fc_c,discard_diagonal=True)


# ## 2.3 Generate Simulated Scatter Plot
# 
# To generate a "cartoonish" scatter plot that exemplifies the non-BOLD behavior--namely points sitting over the identity line--we do the following:
# 
# 1) ```x_data```: We load real FC_c for a pair of echoes (```(e01,e02)```)
# 2) ```y_data```: Instead of loading FC_c for a second pair of echoes, we simulate the FC_c for that second pair of echoes by adding a bit of noise to the one loaded in 1.

# In[18]:


# Compute slope for BOLD dominated behavior
echo_times_dict = TES_MSEC['discovery']
slope = (echo_times_dict['e02'] * echo_times_dict['e03']) / (echo_times_dict['e01'] * echo_times_dict['e02'])

# Real data
x_data            = FC_c[('e01','e02')]
x_data_label      = 'C(TEi,TEj)'
x_data_label_plot =  r"$C(TE_{i},TE_{j})$"

# Simulated data
y_data            = (slope * x_data) + 0.1 * (np.random.normal(0,.25,25425))
y_data_label      = 'C(TEk,TEl)'
y_data_label_plot =  r"$C(TE_{k},TE_{l})$"


# Next, we load the two sets of FC_c into a single pandas dataframe for plotting purposes

# In[19]:


bold_sim_df = pd.DataFrame([x_data,y_data], 
                  index=[x_data_label,y_data_label]).T


# In[20]:


bold_sim_scatter = (
    bold_sim_df.hvplot.scatter(
        x=x_data_label,
        y=y_data_label,
        aspect="square",
        color="black",
        datashade=True,  # if this errors on mpl, switch to rasterize=True
        xlabel=x_data_label_plot,
        ylabel=y_data_label_plot,
        xlim=(-0.6, 0.6),
        ylim=(-0.6, 0.6),
        title="(c) Simulation",
        width=250
    )
    .opts(fontscale=1.0)
)


# ## 2.4. Generate Real Data Scatter Plot
# 
# In this case we load real data for both the x and y axis of the plot

# In[21]:


x_data       = FC_c[('e01','e02')]
x_data_label = f"C({te_labels['e01']},{te_labels['e02']})"
x_data_label_plot =  r"$C( TE_{1},TE_{2})$"

y_data       = FC_c[('e02','e03')]
y_data_label = f"C({te_labels['e02']},{te_labels['e03']})"
y_data_label_plot =  r"$C( TE_{1},TE_{3})$"


# In[22]:


bold_real_df = pd.DataFrame([x_data,y_data], 
                                index=[x_data_label, y_data_label]).T


# In[23]:


bold_real_scatter = (
    bold_real_df.hvplot.scatter(
        x=x_data_label,
        y=y_data_label,
        aspect="square",
        color="black",
        datashade=True,  # if this errors on mpl, switch to rasterize=True
        xlabel=x_data_label_plot,
        ylabel=y_data_label_plot,
        xlim=(-0.6, 0.6),
        ylim=(-0.6, 0.6),
        title="(d) Real Data",
        width=250
    )
    .opts(fontscale=1.0)
)


# # 2.5 Generate additional graphical elements for panels (c) and (d)
# 
# First, we create the reference axes for (0,0)

# In[24]:


zero_marker = (
    hv.VLine(0).opts(color="gray", linestyle="dashed", linewidth=0.5)
    * hv.HLine(0).opts(color="gray", linestyle="dashed", linewidth=0.5)
)


# Second, we create the BOLD line and the text annotations.

# In[25]:


non_BOLD_line = hv.Slope(1, 0).opts(color="red", linestyle="dashed", linewidth=2)

def add_bold_real_label(plot, element):
    ax = plot.handles["axis"]

    # Plain text annotation
    ax.text(
        0.2, 0.05, "y = 1 ⋅ x + 0",
        color="red", fontsize=12, rotation=45, ha="left", va="bottom"
    )

    # LaTeX/mathtext annotation
    ax.text(
        -0.25, -0.1, r"$y = \frac{TE_{2} \cdot TE_{3}}{TE_{1} \cdot TE_{2}} \cdot x + 0$",
        color="green", fontsize=12, ha="left", va="bottom", rotation=75
    )

def add_bold_sim_label(plot, element):
    ax = plot.handles["axis"]

    # Plain text annotation
    ax.text(
        0.2, 0.05, "y = 1 ⋅ x + 0",
        color="red", fontsize=12, rotation=45, ha="left", va="bottom"
    )

    # LaTeX/mathtext annotation
    ax.text(
        -0.25, -0.1, r"$y = \frac{TE_{k} \cdot TE_{l}}{TE_{i} \cdot TE_{j}} \cdot x + 0$",
        color="green", fontsize=12, ha="left", va="bottom", rotation=75
    )

BOLD_line = hv.Slope(slope,0).opts(color='g',linestyle='dashed',linewidth=2)


# Third, we compose the individual panels by adding the non_BOLD_line and the zero marker to each of the scatter plot

# In[26]:


bold_sim_panel  = (bold_sim_scatter  * zero_marker * non_BOLD_line * BOLD_line).opts(hooks=[add_bold_sim_label])
bold_real_panel = (bold_real_scatter * zero_marker * non_BOLD_line * BOLD_line).opts(hooks=[add_bold_real_label])


# Forth, we generate the banner signaling these plots corresponds to BOLD-dominated scenarios

# In[27]:


bold_banner = pn.pane.HTML(
    """
    <div style="
        width:100%;
        background:#000;
        color:#fff;
        font-weight:700;
        text-align:center;
        padding:10px 12px;
        box-sizing:border-box;
        font-size:24px;
    ">
      BOLD Dominated Data
    </div>
    """,
    sizing_mode="scale_width",
)


# ***
# # 3. Compose Final Figure

# In[ ]:


figure01 = pn.Row(pn.Column(non_bold_banner,pn.Row(non_bold_sim_panel,non_bold_real_panel), width=1000),
       pn.Column(bold_banner,    pn.Row(bold_sim_panel,bold_real_panel), width=1000))
figure01.save('./figures/pBOLD_Figure01.html')


# Here is the figure!!!
# 
# ![figure01](figures/pBOLD_Figure01.png)

# 
