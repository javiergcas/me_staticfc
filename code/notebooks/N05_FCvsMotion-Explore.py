# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: BOLD WAVES 2024a
#     language: python
#     name: bold_waves_2024a
# ---

import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import subprocess
import datetime
from utils.basics import PRCS_DATA_DIR, ATLASES_DIR, ATLAS_NAME, PRJ_DIR, CODE_DIR
ATLAS_NAME = 'Power264'
ATLAS_DIR = osp.join(ATLASES_DIR,ATLAS_NAME)
from nilearn.connectome import sym_matrix_to_vec
from sfim_lib.io.afni import load_netcc
from sfim_lib.plotting.fc_matrices import hvplot_fc
import hvplot.pandas
import seaborn as sns
import holoviews as hv
import xarray as xr
import panel as pn
from itertools import combinations_with_replacement, permutations, combinations
from scipy.stats import linregress
import hvplot.xarray

color_map_dict={'White':'#ffffff','Cyan':'#E0FFFF','Orange':'#FFA500','Purple':'#800080',
                'Pink':'#FFC0CB','Red':'#ff0000','Gray':'#808080','Teal':'#008080','Brown':'#A52A2A',
                'Blue':'#0000ff','Yellow':'#FFFF00','Black':'#000000','Pale blue':'#ADD8E6','Green':'#00ff00'}
nw_color_dict = {'Uncertain':'#ffffff',
                 'Sensory/somatomotor Hand':'#E0FFFF',
                 'Sensory/somatomotor Mouth':'#FFA500',
                 'Cingulo-opercular Task Control':'#800080',
                 'Auditory':'#FFC0CB',
                 'Default mode':'#ff0000',
                 'Memory retrieval?':'#808080',
                 'Ventral attention':'#008080',
                 'Visual':'#0000ff',
                 'Fronto-parietal Task Control':'#FFFF00',
                 'Salience':'#000000',
                 'Subcortical':'#A52A2A',
                 'Cerebellar':'#ADD8E6',
                 'Dorsal attention':'#00ff00'}

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# # 1. Load Dataset Information

dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
Nscans          = dataset_info_df.shape[0]
print('++ Number of scans: %s scans' % Nscans)
dataset_scan_list = list(dataset_info_df.index)
Nacqs = 201

# # 2. Load Atlas Information

# +
roi_info_path = osp.join(ATLAS_DIR,f'{ATLAS_NAME}.roi_info.csv')
roi_info_df   = pd.read_csv(roi_info_path)
roi_info_df.head(5)

Nrois = roi_info_df.shape[0]
Ncons = int(((Nrois) * (Nrois-1))/2)

print('++ INFO: Number of ROIs = %d | Number of Connections = %d' % (Nrois,Ncons))
# -

# Compute Euclidean Distance between ROI centroids

# +
# Select the columns that correspond to position
roi_coords_df = roi_info_df.set_index(['ROI_Name'])[['pos_R','pos_A','pos_S']]

# Convert the DataFrame to a NumPy array
roi_coords = roi_coords_df.values

# Calculate the Euclidean distance using broadcasting
roi_distance_matrix = np.sqrt(((roi_coords[:, np.newaxis] - roi_coords) ** 2).sum(axis=2))

# Convert to DataFrame
roi_distance_df = pd.DataFrame(roi_distance_matrix, index=roi_coords_df.index, columns=roi_coords_df.index)
# -

roi_distance_vect = sym_matrix_to_vec(roi_distance_df.values, discard_diagonal=True)

# # 2. Load FC for each scan
#
# ## 2.1 Motion corrected data (within-echo)

fc_all_sbjs_R = {}
fc_all_sbjs_Z = {}

echo_pairs_tuples = [i for i in combinations_with_replacement(['e01','e02','e03'],2)]
echo_pairs        = [('|').join(i) for i in echo_pairs_tuples]
echo_pair_combos  = [c for c in combinations(echo_pairs,2)]
cenmodes          = ['ALL','KILL']
input_types       = ['volreg','MEICA']

fc = xr.DataArray(dims=['scan','input','cenmode','pair','ROI_x','ROI_y'],
                  coords={'scan':['|'.join([sbj,ses]) for (sbj,ses) in dataset_scan_list],
                          'input':input_types,
                          'cenmode':cenmodes,
                          'pair':echo_pairs,
                          'ROI_x':roi_info_df['ROI_Name'].values,
                          'ROI_y':roi_info_df['ROI_Name'].values})

# Load data following motion correct + basic denoising (per echo) 36,37,52

# %%time
for cenmode in cenmodes:
    for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
        for (e_x,e_y) in echo_pairs_tuples:
            roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.volreg.scale.tproject_{cenmode}.{ATLAS_NAME}_000.netts')
            roi_ts_x      = np.loadtxt(roi_ts_path_x)
            roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.volreg.scale.tproject_{cenmode}.{ATLAS_NAME}_000.netts')
            roi_ts_y      = np.loadtxt(roi_ts_path_y)
            aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
            aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
            # Compute the full correlation matrix between aux_ts_x and aux_ts_y
            aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
            fc.loc['|'.join([sbj,ses]),'volreg',cenmode,'|'.join([e_x,e_y]),:,:] = aux_r

# Load data following MEICA denosing (per echo)

# %%time
for cenmode in cenmodes:
    for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
        for (e_x,e_y) in echo_pairs_tuples:
            roi_ts_path_x = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_x}.meica_dn.scale.tproject_{cenmode}.{ATLAS_NAME}_000.netts')
            roi_ts_x      = np.loadtxt(roi_ts_path_x)
            roi_ts_path_y = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}',f'errts.{sbj}.r01.{e_y}.meica_dn.scale.tproject_{cenmode}.{ATLAS_NAME}_000.netts')
            roi_ts_y      = np.loadtxt(roi_ts_path_y)
            aux_ts_x = pd.DataFrame(roi_ts_x, columns=roi_info_df['ROI_Name'].values)
            aux_ts_y = pd.DataFrame(roi_ts_y, columns=roi_info_df['ROI_Name'].values)
            # Compute the full correlation matrix between aux_ts_x and aux_ts_y
            aux_r   = np.corrcoef(aux_ts_x.T, aux_ts_y.T)[:aux_ts_x.shape[1], aux_ts_x.shape[1]:]
            fc.loc['|'.join([sbj,ses]),'MEICA',cenmode,'|'.join([e_x,e_y]),:,:] = aux_r

# +
# %%time
linefits = xr.DataArray(dims=['scan','input','cenmode','pair_combo','metric'],
                        coords={'scan':['|'.join([sbj,ses]) for (sbj,ses) in dataset_scan_list],
                          'input':input_types,
                          'cenmode':cenmodes,
                          'pair_combo':['-'.join([x,y]) for x,y in echo_pair_combos],
                          'metric':['Slope','Intercept','R','p_val']})

for scan in tqdm(fc.coords['scan']):
    for input_type in fc.coords['input']:
        for cenmode in fc.coords['cenmode']:
            for ep_x,ep_y in echo_pair_combos:
                this_fc_px = sym_matrix_to_vec(fc.sel(scan=scan,input=input_type,cenmode=cenmode,pair=ep_x).values,discard_diagonal=True)
                this_fc_py = sym_matrix_to_vec(fc.sel(scan=scan,input=input_type,cenmode=cenmode,pair=ep_y).values,discard_diagonal=True)
                data = pd.DataFrame(np.vstack([this_fc_px,this_fc_py]).T,columns = [ep_x,ep_y])
                slope,intercept,R,p_val,_ = linregress(data)
                linefits.loc[scan,input_type,cenmode,'-'.join([ep_x,ep_y]),'Slope'] = slope
                linefits.loc[scan,input_type,cenmode,'-'.join([ep_x,ep_y]),'Intercept'] = intercept
                linefits.loc[scan,input_type,cenmode,'-'.join([ep_x,ep_y]),'R'] = R
                linefits.loc[scan,input_type,cenmode,'-'.join([ep_x,ep_y]),'p_val'] = p_val
# -

(pd.DataFrame(linefits.sel(input='volreg',cenmode='ALL',metric='Slope').mean(dim='pair_combo').values).hvplot.kde(label='Basic Denoising', xlabel='Slope') * \
pd.DataFrame(linefits.sel(input='MEICA',cenmode='ALL',metric='Slope').mean(dim='pair_combo').values).hvplot.kde(label='MEICA Denoising') * \
hv.VLine(1).opts(line_color='k',line_dash='dashed') )

pd.DataFrame(linefits.sel(input='volreg',cenmode='ALL',metric='Intercept').mean(dim='pair_combo').values).hvplot.kde(label='Basic Denoising', xlabel='Intercept') * \
pd.DataFrame(linefits.sel(input='MEICA',cenmode='ALL',metric='Intercept').mean(dim='pair_combo').values).hvplot.kde(label='MEICA Denoising') * \
hv.VLine(0).opts(line_color='k',line_dash='dashed')

pd.DataFrame(linefits.sel(input='volreg',cenmode='KILL',metric='Slope').mean(dim='pair_combo').values).hvplot.kde(label='Basic Denoising', xlabel='Slope') * \
pd.DataFrame(linefits.sel(input='MEICA',cenmode='KILL',metric='Slope').mean(dim='pair_combo').values).hvplot.kde(label='MEICA Denoising') * \
hv.VLine(1).opts(line_color='k',line_dash='dashed')

pd.DataFrame(linefits.sel(input='volreg',cenmode='KILL',metric='Intercept').mean(dim='pair_combo').values).hvplot.kde(label='Basic Denoising', xlabel='Intercept') * \
pd.DataFrame(linefits.sel(input='MEICA',cenmode='KILL',metric='Intercept').mean(dim='pair_combo').values).hvplot.kde(label='MEICA Denoising') * \
hv.VLine(0).opts(line_color='k',line_dash='dashed')

scan_select    = pn.widgets.Select(name='Scan', options=['|'.join([sbj,ses]) for (sbj,ses) in dataset_scan_list],width=150)
input_select   = pn.widgets.Select(name='Input', options=input_types,width=150)
cenmode_select = pn.widgets.Select(name='Input', options=cenmodes,width=150)
fcsel_card     = pn.Card(scan_select, input_select, cenmode_select, title='FC Selection')
echo_pair_x_select = pn.widgets.Select(name='Echo Pair 1',options=echo_pairs,width=150, value=echo_pairs[0])
echo_pair_y_select = pn.widgets.Select(name='Echo Pair 2',options=echo_pairs,width=150, value=echo_pairs[1])
echo_pair_card = pn.Card(echo_pair_x_select, echo_pair_y_select, title='Scatter Selection')


@pn.depends(scan_select,input_select,cenmode_select)
def plot_fc_all_pairs(scan,input,cenmode):
    out = pn.GridBox(ncols=3)
    for (e_x,e_y) in echo_pairs_tuples:
        this_fc    = fc.sel(scan=scan,input=input,cenmode=cenmode,pair='|'.join([e_x,e_y]))
        this_fc_df = pd.DataFrame(this_fc.values,
                                  index=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index,
                                  columns=roi_info_df.set_index(['ROI_Name','ROI_ID','Hemisphere','Network','RGB']).index)
        this_fc_plot = hvplot_fc(this_fc_df,major_label_overrides='regular_grid',cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', net_cmap=nw_color_dict).opts(title=e_x+' vs. '+e_y)
        out.append(this_fc_plot)
    return out


@pn.depends(scan_select,input_select,cenmode_select,echo_pair_x_select,echo_pair_y_select)
def plot_fc_scatters(scan,input,cenmode,ep_x,ep_y):
    this_fc_px = sym_matrix_to_vec(fc.sel(scan=scan,input=input,cenmode=cenmode,pair=ep_x).values,discard_diagonal=True)
    this_fc_py = sym_matrix_to_vec(fc.sel(scan=scan,input=input,cenmode=cenmode,pair=ep_y).values,discard_diagonal=True)
    data = pd.DataFrame(np.vstack([this_fc_px,this_fc_py]).T,columns = [ep_x,ep_y])
    slope,intercept,R,p_val,_ = linregress(data)
    
    plot = data.hvplot.scatter(x=ep_x,y=ep_y,aspect='square',datashade=True, frame_width=400,xlim=(-1,1),ylim=(-1,1),color='black',fontsize={'labels':16})
    plot = plot * hv.Slope(1,0).opts(line_dash='dashed', line_color='gray', line_width=1)
    plot = plot * hv.Slope(slope,intercept).opts(line_color='blue', line_width=1)
    info_text = 'y=%.2f+%.2f*x | R2=%.2f' % (intercept,slope,np.power(R,2))
    plot = plot * hv.Text(0,.8, info_text)
    return plot


dashboard = pn.Row(pn.Column(fcsel_card,echo_pair_card),plot_fc_all_pairs,plot_fc_scatters).show(port=port_tunnel,open=False)

dashboard.stop()

# # 3. Load Mean Motion Framewise Displacement
#
# According to [Power et al. "Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion"](https://www.sciencedirect.com/science/article/pii/S1053811911011815)
#
# #### Framewise displacement (FD) calculations
#
# "Differentiating head realignment parameters across frames yields a six dimensional timeseries that represents instantaneous head motion. To express instantaneous head motion as a scalar quantity we used the empirical formula, FDi = |Δdix| + |Δdiy| + |Δdiz| + |Δαi| + |Δβi| + |Δγi|, where Δdix = d(i − 1)x − dix, and similarly for the other rigid body parameters [dix diy diz αi βi γi]. Rotational displacements were converted from degrees to millimeters by calculating displacement on the surface of a sphere of radius 50 mm, which is approximately the mean distance from the cerebral cortex to the center of the head."

#mot_mean_fd = np.empty((Nscans,Ncons))
mot_mean_fd_DF = pd.DataFrame(index=dataset_info_df.index,columns=['Mean FD'])
for i,(sbj,ses) in enumerate(tqdm(dataset_scan_list)):
    mot_path = osp.join(PRCS_DATA_DIR,sbj,f'D02_Preproc_fMRI_{ses}','dfile.r01.1D')
    # Load the motion parameters
    motion_params = np.loadtxt(mot_path)
    # Calculate the differences between consecutive time points
    diffs = np.diff(motion_params, axis=0)
    # Optionally convert rotational parameters (in radians) to mm (assuming a brain radius of 50 mm)
    brain_radius = 0.050  # in mm
    diffs[:, 3:] *= brain_radius
    # Compute framewise displacement (FD)
    FD = np.sum(np.abs(diffs), axis=1)
    # Pad with 0 for the first time point (no preceding frame to compare with)
    #FD = np.insert(FD, 0, 0)
    mot_mean_fd_DF.loc[(sbj,ses)] = FD.mean()
mot_mean_fd_DF = mot_mean_fd_DF.infer_objects()

mot_mean_fd_DF.sort_values('Mean FD')


