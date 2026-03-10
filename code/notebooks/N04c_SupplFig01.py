#!/usr/bin/env python
# coding: utf-8

# # Description: Supplementary Figure 01
# 
# This notebook exemplifies the issue of heteroscedasticity and how it could affect more straight forward computations of pBOLD

# In[1]:


import pandas as pd
import numpy as np
import holoviews as hv
import panel as pn

import os.path as osp
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, TES_MSEC

ATLAS_NAME = 'Power264-discovery'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)

import hvplot.pandas
from nilearn.connectome import sym_matrix_to_vec

from utils.basics import echo_pairs_tuples, get_altas_info

echo_times_dict = TES_MSEC['discovery']


# *** 
# # 1. Load Parcellation Information
# 
# This is needed to plot the two connectivity matrices in panels (a) and (b)

# In[2]:


roi_info_df, power264_nw_cmap = get_altas_info(ATLAS_DIR,ATLAS_NAME)
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# ***
# # 2. Load Timeseries and FC_C across alll echoes for a representative scan
# 
# We will use a low motion and aggresively pre-processed scan for explanatory purposes.

# In[3]:


sbj = 'MGSBJ05'
ses = 'constant_gated'
scenario = 'ALL_Tedana-fastica'


# In[4]:


fc={}
for (e_x,e_y) in echo_pairs_tuples:
    roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.{e_x}.volreg.spc.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_x      = np.loadtxt(roi_ts_path_x)
    roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D03_Preproc_{ses}_NORDIC-off',f'errts.{sbj}.r01.{e_y}.volreg.spc.tproject_{scenario}.{ATLAS_NAME}_000.netts')
    roi_ts_y      = np.loadtxt(roi_ts_path_y)
    aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
    aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
    # Compute the full correlation matrix between aux_ts_x and aux_ts_y
    aux_c           = np.cov(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
    fc[(e_x,e_y)]  = pd.DataFrame(aux_c,index=roi_idxs,columns=roi_idxs)


# In[5]:


x_te1,x_te2    = 'e01','e02'
y_te1,y_te2    = 'e02','e03'


# In[6]:


# Get Connectivity data: top triangle of the FC-C matrix
a = sym_matrix_to_vec(fc[(x_te1,x_te2)].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc[(y_te1,y_te2)].values,discard_diagonal=True)
df = pd.DataFrame([a,b], index=['C(TEi,TEj)','C(TEk,TEl)']).T
x = df['C(TEi,TEj)'].values
y = df['C(TEk,TEl)'].values

df = pd.DataFrame([a,b], index=['C(TE1,TE2)','C(TE2,TE3)']).T


# ***
# # 2. Create Suppl. Figure 01 - Panel A
# 
# Create lines marking the origin

# In[7]:


hv.extension("matplotlib")
zero_point = hv.HLine(0).opts(linewidth=0.5, color='k', linestyle='dotted') * hv.VLine(0).opts(linewidth=0.5, color='k', linestyle='dotted')


# Create line representing non-BOLD behavior (slope=1, intercept=0)

# In[8]:


nonBOLD_line = hv.Slope(1,0).opts(linewidth=2, color='r', linestyle='dashed')


# Create line representing BOLD behavior (slope=f(echo times), intercept=0)

# In[9]:


BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])
BOLD_line = hv.Slope(BOLD_slope,0).opts(linewidth=2, color='g', linestyle='dashed') 


# Create line that best fits the data

# In[10]:


data_fit = np.polyfit(a,b,1)
data_line = hv.Slope(data_fit[0],data_fit[1]).opts(linewidth=2, color='b', linestyle='dashed') 


# Create scatter plot for FCc at two representative echo time pairs.

# In[11]:


scat_datashaded = df.hvplot.scatter(x='C(TE1,TE2)',y='C(TE2,TE3)', 
                                    xlabel=r"$FC_{c}(TE_{1},TE_{2})$",
                                    ylabel=r"$FC_{c}(TE_{2},TE_{3})$",
                                    aspect='square', xlim=(-.1,.5), ylim=(-.1,.5), datashade=True, frame_width=400, fontscale=1.1).opts(title='(a) Real Data')


# Put it all together and add text annotations

# In[12]:


def add_text(plot, element):
    ax = plot.handles["axis"]

    # Non-BOLD Line text
    ax.text(
        0.3, 0.25, r"$y = 1 \cdot x + 0.0$",
        color="red", fontsize=12, rotation=45, ha="left", va="bottom"
    )

    # BOLD Line text
    ax.text(
        0.01, 0.20, r"$ y = %.1f \cdot x + 0.0$" % BOLD_slope,
        color="green", fontsize=12, ha="left", va="bottom", rotation=75
    )

    # Linear Fit text
    ax.text(
        0.14, 0.26, r"$ y = %.1f \cdot x + %.1f$" % (data_fit[0],data_fit[1]),
        color="blue", fontsize=12, ha="left", va="bottom", rotation=63
    )


# In[13]:


panel_a = (scat_datashaded * nonBOLD_line * BOLD_line * data_line * zero_point).opts(hooks=[add_text])


# ***
# # 3. Create Supp. Figure 01 - Panel b: LOWESS Plot

# In[14]:


from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

X = a
# Add a constant for the intercept
X_const = sm.add_constant(X)
model = sm.OLS(b, X_const).fit()

# Run Breusch-Pagan test
# The function takes residuals and design matrix (exog)
test_stat, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)

print(f"Breusch-Pagan test statistic: {test_stat:.3f}")
print(f"P-value: {p_value:.3f}")


# In[15]:


df2=pd.DataFrame([model.fittedvalues,model.resid], index=['Fitted','Residuals']).T


# In[16]:


from statsmodels.nonparametric.smoothers_lowess import lowess

# Compute LOWESS smoother
smoothed = lowess(df2['Residuals'], df2['Fitted'], frac=0.3)
smooth_df = pd.DataFrame(smoothed, columns=['Fitted', 'Residuals'])

# Scatter plot of residuals
scatter = df2.hvplot.scatter(x='Fitted', y='Residuals', alpha=0.7, color='k', datashade=True, ylim=(-.1,.2), xlim=(-.1,.2), fontscale=1.1, frame_width=650)

# Smoothed curve
smooth_curve = smooth_df.hvplot.line(x='Fitted', y='Residuals', color='red')

# Horizontal zero line
zero_line = hv.HLine(0).opts(color='black', linestyle='dashed', linewidth=1.0)

# Combine plots
panel_b = (scatter * smooth_curve * zero_line).opts(
    title='(b) Fitted Values vs. Residuals & LOWESS line',
    xlabel='Fitted values', ylabel='Residuals')


# In[17]:


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
        width:1675px;
    ">
      Issue with quantification via linear fit
    </div>
    """,
    sizing_mode="scale_width",
)


# In[18]:


SuppFig01 = pn.Column(bold_banner, pn.Row(panel_a,panel_b))
SuppFig01.save('./figures/pBOLD_SuppFig01.html')


# Here is the final suppl. figure 01 in static form (for github rendering)
# 
# ![Suppl. Figure 01](figures/pBOLD_SuppFig01.png)
