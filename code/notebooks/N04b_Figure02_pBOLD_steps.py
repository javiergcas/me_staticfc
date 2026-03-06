#!/usr/bin/env python
# coding: utf-8

# # Description: pBOLD computation step-by-step
# 
# This notebook can be used to generate Figure 02 in the accompanying publication. This figure contains information about how pBOLD is computed given a multi-echo dataset.

# In[ ]:


import pandas as pd
import numpy as np
import holoviews as hv
import os.path as osp
import os
from tqdm import tqdm
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, PRJ_DIR, CODE_DIR, TES_MSEC
from utils.basics import power264_nw_cmap
from utils.fc_matrices import hvplot_fc
import matplotlib.colors as mcolors

#from sfim_lib.plotting.fc_matrices import hvplot_fc
from itertools import combinations_with_replacement, combinations
ATLAS_NAME = 'Power264-discovery'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
import panel as pn
pn.extension('mathjax')   # panel comms + latex support
hv.extension('bokeh')     # set backend early
from nilearn.connectome import sym_matrix_to_vec

from utils.basics import echo_pairs_tuples, echo_pairs, pairs_of_echo_pairs
from utils.basics import chord_distance_between_intersecting_lines, line_circle_intersection

echo_times_dict = TES_MSEC['discovery']


# *** 
# # 1. Load Parcellation Information
# 
# This is needed to plot the two connectivity matrices in panels (a) and (b)

# In[77]:


roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df.head(5)

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)

print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))
roi_idxs = roi_info_df.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index


# ***
# # 2. Load Timeseries and FC_C across alll echoes for a representative scan [Steps 1 - 3]
# 
# We will use a low motion and aggresively pre-processed scan for explanatory purposes.

# In[79]:


sbj = 'MGSBJ05'
ses = 'constant_gated'
scenario = 'ALL_Tedana-fastica'


# In[80]:


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


# In[81]:


from utils.basics import compute_residuals, project_points
from bokeh.models import FixedTicker

x_te1,x_te2    = 'e01','e02'
y_te1,y_te2    = 'e02','e03'


# In[82]:


# Get Connectivity data: top triangle of the FC-C matrix
a = sym_matrix_to_vec(fc[(x_te1,x_te2)].values,discard_diagonal=True)
b = sym_matrix_to_vec(fc[(y_te1,y_te2)].values,discard_diagonal=True)
df = pd.DataFrame([a,b], index=['C(TEi,TEj)','C(TEk,TEl)']).T
x = df['C(TEi,TEj)'].values
y = df['C(TEk,TEl)'].values


# In[83]:


data        = fc[(x_te1,x_te2)]
plot_fc_x   = hvplot_fc(data,major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
         cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left',clim=(-.3,.3),
         cbar_title=r"$$FC_{c}(TE_{i},TE_{j})$$", cbar_title_fontsize=14, ticks_font_size=14).opts(tools=[], active_tools=[],toolbar=None)
data        = fc[(y_te1,y_te2)]
plot_fc_y   = hvplot_fc(data,major_label_overrides='regular_grid', net_cmap=power264_nw_cmap,
         cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left',clim=(-.3,.3),
         cbar_title=r"$$FC_{c}(TE_{k},TE_{l})$$", cbar_title_fontsize=14, ticks_font_size=14).opts(tools=[],active_tools=[], toolbar=None)
pn.Row(plot_fc_x,plot_fc_y).save('./figures/pBOLD_Figure02_ab.html')


# Here are the FC matrices, saved as HTML on the prior cell, saved as PNG for rendering in github:
# 
# ![Figure 02 Panels a and b](figures/pBOLD_Figure02_ab.png)

# Next, we plot panel (c), which is a scatter plot of the top triangle of these two FC_c matrices against each other

# In[84]:


# Create basic graphic elements: zero point, BOLD line, non-BOLD line, basic scatter, etc.
zero_point         = hv.HLine(0).opts(line_width=0.5, line_color='k', line_dash='dotted') * hv.VLine(0).opts(line_width=0.5, line_color='k', line_dash='dotted')
fcc_scat_overlap   = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), rasterize=True, frame_width=400,
                                        cmap='cividis', 
                                        xlabel=r"$$FC_{c}(TE_{i},TE_{j})$$", 
                                        ylabel=r"$$FC_{c}(TE_{k},TE_{l})$$",
                                        fontscale=1.3).opts(colorbar_opts={"title": "Num. Overlapping Edges:", 'major_label_text_font_size':12},toolbar=None, active_tools=['reset'])
pn.Row(zero_point * fcc_scat_overlap).save('./figures/pBOLD_Figure02_c.html')


# Here is a static version of panel (c) saved in the cell above for github rendering purposes
# 
# ![Figure 02 Panel c](figures/pBOLD_Figure02_c.png)

# ***
# # 3. Distance to line indicating S0 dominated data [Step 4]
# 
# We first define the line that exemplifies expected behavior when data in dominated by non-BOLD effects (slope = 1, intercept=0)

# In[85]:


So_slope     = 1.0
nonBOLD_line = hv.Slope(1,0).opts(line_width=3, line_color='r', line_dash='dashed')


# Next, we define the line that exemplifies the behavior when data is dominated by BOLD effects (slope = f(contributing echoes), intercept=0)

# In[86]:


BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])
BOLD_line  = hv.Slope(BOLD_slope,0).opts(line_width=3, line_color='g', line_dash='dashed')


# Third, we select a connection (one dot on the scatter plot) in the vicinity of 0.15, 0.25. This is a good location for illustrative purposes becuase it sits in between both BOLD and non-BODL lines

# In[87]:


# Let's select a representative connection
# We pick on around 0.15, 0.25 becuase it is easy to visualize

x_target, y_target = 0.15,0.25

# Compute Euclidean distance
distances = np.sqrt((df["C(TEi,TEj)"] - x_target)**2 + (df["C(TEk,TEl)"] - y_target)**2)

# Get index of closest row
closest_idx = distances.idxmin()

# Extract the row
closest_row = df.loc[closest_idx]

sample_x,sample_y =  closest_row.values
sample_point = hv.Points((sample_x,sample_y)).opts(size=10,color='k', marker='x')
print(closest_row, sample_x, sample_y)


# Forth, we compute the location of this connection when projected over the non-BOLD line, as well as the segment joining the connection and its non-BOLD projection

# In[88]:


proj_on_So_line             = project_points(sample_x,sample_y,So_slope,0.0)
proj_on_So_line_point       = hv.Points(proj_on_So_line).opts(size=10,color='r', marker='s')
proj_on_So_residual_segment = hv.Path([[(sample_x,sample_y), proj_on_So_line]]).opts(color="red", line_width=2)


# Fifth, we put all the elements together to generate Panel d

# In[89]:


panel_d = pn.Row((fcc_scat_overlap * zero_point) * BOLD_line * nonBOLD_line * sample_point * proj_on_So_line_point * proj_on_So_residual_segment)


# Panel e shows the same information as panel d, except that connections are colored according to their distance to the So line. To generate this additional panel, we need to compute these distances and add then to the pandas Dataframe

# In[90]:


df['dSo'] = compute_residuals(df["C(TEi,TEj)"].values,df["C(TEk,TEl)"].values,1.0,0.0)


# In[91]:


fcc_scat_dSo   = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), frame_width=400,
                                        cmap='viridis', s=0.75, color='dSo',
                                        xlabel=r"$$FC_{c}(TE_{i},TE_{j})$$", 
                                        ylabel=r"$$FC_{c}(TE_{k},TE_{l})$$",
                                        fontscale=1.3).opts(title='',toolbar=None,
                                                            clim=(df['dSo'].quantile(0.01),df['dSo'].quantile(0.95)),
                                                            colorbar_opts={"title": "Distance to So line:"})

panel_e = fcc_scat_dSo * zero_point * nonBOLD_line


# In[92]:


pn.Row(panel_d,panel_e).save('./figures/pBOLD_Figure02_de.html')


# Here are panels d and e in static form for github rendering purposes
# 
# ![Figure 02 panel d and e](figures/pBOLD_Figure02_de.png)

# ***
# # 4. Distance to the line indicating BOLD dominated data [Step 5]
# 
# This is similar to the previous section, but the reference here is the line associated with BOLD dominated behavior (green line)

# In[93]:


proj_on_BOLD_line             = project_points(sample_x,sample_y,BOLD_slope,0.0)
proj_on_BOLD_line_point       = hv.Points(proj_on_BOLD_line).opts(size=10,color='g', marker='s')
proj_on_BOLD_residual_segment = hv.Path([[(sample_x,sample_y), proj_on_BOLD_line]]).opts(color="green", line_width=2)


# In[94]:


panel_f = pn.Row((fcc_scat_overlap * zero_point) * BOLD_line * nonBOLD_line * sample_point * proj_on_BOLD_line_point * proj_on_BOLD_residual_segment)


# In[95]:


df['dBOLD'] = compute_residuals(df["C(TEi,TEj)"].values,df["C(TEk,TEl)"].values,BOLD_slope,0.0)


# In[96]:


fcc_scat_dBOLD   = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), frame_width=400,
                                        cmap='viridis', s=0.75, color='dBOLD',
                                        xlabel=r"$$FC_{c}(TE_{i},TE_{j})$$", 
                                        ylabel=r"$$FC_{c}(TE_{k},TE_{l})$$",
                                        fontscale=1.3).opts(title='',toolbar=None,
                                                            clim=(df['dBOLD'].quantile(0.01),df['dBOLD'].quantile(0.95)),
                                                            colorbar_opts={"title": "Distance to BOLD line:"})

panel_g = fcc_scat_dBOLD * zero_point * BOLD_line


# In[97]:


pn.Row(panel_f,panel_g).save('./figures/pBOLD_Figure02_fg.html')


# Here are panels f and g in statit form for github rendering purposes
# 
# ![Figure 02 Panels f and g](figures/pBOLD_Figure02_fg.png)

# ***
# # 5. Weigthed preference towards the BOLD like [Steps 6 - 7]
# 
# ## 5.1. [Step 6]: Compute unweigthed BOLD line preference
# 
# First, we will generate panel h, which shows what BOLD preference value (-1 = Prefernece toward non-BOLD, 1 = Preference towards BOLD, 0.5 = Uncertain) is assigned to each connection.

# In[98]:


# Let's now see which line is the closest to each connection.
# When the difference in distance is below the tolerance, we call it a tie and assign a 0.5
pref1 = (df['dBOLD'] < df['dSo']).astype(float)
ties = np.isclose(df['dBOLD'], df['dSo'], atol=1e-3)
pref1[ties] = 0.5
df['prefBOLD'] = pref1


# In[99]:


ticks = [0.25, 0.5, 0.75]
labels = {0.25: "non-BOLD", 0.5:"Uncertain", 0.75: "BOLD"}
scat_colored_by_prefBOLD = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), frame_width=400,
                                            size=.75, color='prefBOLD', 
                                            cmap=['#FF7F7F','#FFFF7F','lightgreen'],
                                            xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", 
                                            ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$",
                                            fontscale=1.3).opts(title='',
                                                                clim=(df['dBOLD'].quantile(0.01),df['prefBOLD'].quantile(0.95)),
                                                                colorbar_opts={"title": "Line preference (e.g., likely behavior):","ticker": FixedTicker(ticks=ticks),
                                                                                "major_label_overrides": labels,}, toolbar=None)


# In[100]:


panel_h = scat_colored_by_prefBOLD * zero_point * BOLD_line * nonBOLD_line


# ## 5.2. [Step 7]: Compute distance to origin and weigth line preference by such distances
# 
# First, we define the weight function, which in this case is simply the distance to the origin. Other options could be the square of that, its square root, etc.
# 
# We also define a limit, to avoid giving excesive weigth to isolated connections with excessively high covariance values

# In[101]:


weight_fn     = lambda r: np.power(r,1.0)
max_weight_fn = lambda r: np.minimum(r,np.quantile(r,.95))


# Estimating the distance to the origin is as simple as the sqaure root of the sum of the x and y coordinates squared.

# In[102]:


r  = np.sqrt(x**2 + y**2)


# The next code applies the weight function (in this case it does nothing), and then the limiting function

# In[103]:


if weight_fn is None:
    weight_fn = lambda r: r   # linear weight by radius
w = weight_fn(r)
if max_weight_fn is not None:
    w = max_weight_fn(w)
total_weight = w.sum()


# We add the weights to a new column to our dataframe

# In[104]:


df['w'] = w


# We are not ready to generate panel i, the scatter with the connections colored by the distance to the origin

# In[105]:


scat_colored_by_w = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), frame_width=400,
                                    size=.75, color='w', 
                                    cmap='gray_r',
                                    xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", 
                                    ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$", 
                                    fontscale=1.3).opts(title='',
                                                        clim=(df['w'].quantile(0.05),df['w'].quantile(0.95)),
                                                        colorbar_opts={"title": "Weights [Distance to origin]:"},
                                                        toolbar=None)
panel_i = scat_colored_by_w * zero_point * BOLD_line * nonBOLD_line


# Finally, we multiply the original BOLD-line preference by our final weight values in order to obtain the weighted BOLD line preference for each connection.

# In[106]:


weighted_pref1 = (w * pref1)
frac_line1 = weighted_pref1
df['w_prefBOLD'] = frac_line1


# In[107]:


# We create a continuous map that goes through the same colors used to plot the categorical preference values in oabel h 
colors = ['#FF7F7F','#FFFF7F','lightgreen']
cmap = mcolors.LinearSegmentedColormap.from_list("green_yellow_red", colors)

scat_colored_by_wprefBOLD = df.hvplot.scatter(x='C(TEi,TEj)',y='C(TEk,TEl)', aspect='square', xlim=(-.1,.4), ylim=(-.1,.4), frame_width=400,
                                              size=.75, color='w_prefBOLD', 
                                              cmap=cmap,
                                              xlabel=r"$$FC_{C}(TE_{i},TE_{j})$$", 
                                              ylabel=r"$$FC_{C}(TE_{k},TE_{l})$$", 
                                              fontscale=1.3).opts(title='',
                                                                  clim=(df['w_prefBOLD'].quantile(0.05),df['w_prefBOLD'].quantile(0.95)),
                                                                  colorbar_opts={"title": "Weigthed Pref. towards BOLD line:"},
                                                                  toolbar=None)
panel_j = scat_colored_by_wprefBOLD * zero_point * BOLD_line * nonBOLD_line


# In[108]:


pn.Row(panel_h,panel_i,panel_j).save('./figures/pBOLD_Figure02_hij.html')


# Here are panels h-j in static form for github rendering
# 
# ![Figure 02 Panels h,i,j](figures/pBOLD_Figure02_hij.png)

# ***
# 
# # 6. Using the Chord distance to weight averate pBOLD obtained for the different TE quadruples
# 
# Here we only have the code to exemplify how the chord distance is used while computing pBOLD. This is not a complete implementation of the pBOLD computation. 
# 
# For that, please check the program ```python/compute_pBOLD.py``` available as part of this repo.

# First, we look at a case where the BOLD and non-BOLD lines are further apart

# In[109]:


# CASE 1:
x_te1,x_te2='e01','e03'
y_te1,y_te2='e03','e03'
BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])
BOLD_line  = hv.Slope(BOLD_slope,0).opts(line_width=3, line_color='g', line_dash='dashed') 

# Create an Ellipse element with center (0,0) and diameter 2 (radius 1)
unit_circle            = hv.Ellipse(0, 0, 1).opts(line_dash='dashed')
BOLD_point_on_circle   = np.array(line_circle_intersection(BOLD_slope,0.0,r=0.5))
So_point_on_circle     = np.array(line_circle_intersection(So_slope,0.0,r=0.5))
BOLD_dot               = hv.Points([BOLD_point_on_circle]).opts(size=5,color='g')
So_dot                 = hv.Points([So_point_on_circle]).opts(size=5,color='r')
BOLD2So_path           = hv.Path([[BOLD_point_on_circle, So_point_on_circle]]).opts(color="black", line_width=2)
segment                = np.sqrt((So_point_on_circle-BOLD_point_on_circle)**2)/2
BOLD2So_path_middle    = np.array([So_point_on_circle[0]-segment[0],So_point_on_circle[1]+segment[1]])
line_with_dist = hv.Arrow(BOLD2So_path_middle[0],BOLD2So_path_middle[1],'%.2f' % chord_distance_between_intersecting_lines(1.0, BOLD_slope, r=0.5),'^')

panel_k = (fcc_scat_overlap.opts(colorbar=False) * BOLD_line * nonBOLD_line * zero_point * unit_circle * BOLD_dot * So_dot * BOLD2So_path * line_with_dist).opts(xlim=(-.51,.51), ylim=(-.51,.51), aspect='square', fontscale=1.3, 
                                                                                                                          xlabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[x_te1],echo_times_dict[x_te2]), 
                                                                                                                          ylabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[y_te1],echo_times_dict[y_te2]))


# Second, we look at a case where both lines are more close to each other

# In[110]:


# CASE 2:
x_te1,x_te2='e02','e02'
y_te1,y_te2='e02','e03'
BOLD_slope = (echo_times_dict[y_te1]*echo_times_dict[y_te2])/(echo_times_dict[x_te1]*echo_times_dict[x_te2])
BOLD_line  = hv.Slope(BOLD_slope,0).opts(line_width=3, line_color='g', line_dash='dashed') 

# Create an Ellipse element with center (0,0) and diameter 2 (radius 1)
unit_circle            = hv.Ellipse(0, 0, 1).opts(line_dash='dashed')
BOLD_point_on_circle   = np.array(line_circle_intersection(BOLD_slope,0.0,r=0.5))
So_point_on_circle     = np.array(line_circle_intersection(So_slope,0.0,r=0.5))
BOLD_dot               = hv.Points([BOLD_point_on_circle]).opts(size=5,color='g')
So_dot                 = hv.Points([So_point_on_circle]).opts(size=5,color='r')
BOLD2So_path           = hv.Path([[BOLD_point_on_circle, So_point_on_circle]]).opts(color="black", line_width=2)
segment                = np.sqrt((So_point_on_circle-BOLD_point_on_circle)**2)/2
BOLD2So_path_middle    = np.array([So_point_on_circle[0]-segment[0],So_point_on_circle[1]+segment[1]])
line_with_dist = hv.Arrow(BOLD2So_path_middle[0],BOLD2So_path_middle[1],'%.2f' % chord_distance_between_intersecting_lines(1.0, BOLD_slope, r=0.5),'^')

panel_l = (fcc_scat_overlap.opts(colorbar=False) * BOLD_line * nonBOLD_line * zero_point * unit_circle * BOLD_dot * So_dot * BOLD2So_path * line_with_dist).opts(xlim=(-.51,.51), ylim=(-.51,.51), aspect='square', fontscale=1.3, 
                                                                                                                          xlabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[x_te1],echo_times_dict[x_te2]), 
                                                                                                                          ylabel=r"$$FC_{C}$$"+ "(%.1f,%.1f)"%(echo_times_dict[y_te1],echo_times_dict[y_te2]))


# In[111]:


pn.Row(panel_k,panel_l).save('./figures/pBOLD_Figure02_kl.html')


# ![Figure 02 panels k and l](figures/pBOLD_Figure02_kl.png)
