#!/usr/bin/env python
# coding: utf-8

# # Description: Generate Suppl. Figure 4

# In[1]:


import pandas as pd
import numpy as np
import os.path as osp
from utils.basics import get_dataset_index
from utils.basics import PRCS_DATA_DIR
from tqdm.notebook import tqdm
import panel as pn
pn.extension()


# In[2]:


DATASET = 'evaluation'
ds_index = get_dataset_index(DATASET)
ses_list = list(ds_index.get_level_values('Session').unique())
sbj_list = list(ds_index.get_level_values('Subject').unique())
dataset_info_df = pd.DataFrame(index=pd.MultiIndex.from_product([sbj_list,ses_list],names=['Subject','Session']))


# Load Thermal Noise estimates for the evaluation dataset

# In[3]:


df_list = []
for sbj,ses in tqdm(dataset_info_df.index, desc='Scan'):
    for e in range(1,4):
        aux_path = osp.join(PRCS_DATA_DIR,sbj,'D02_NORDIC',f'{sbj}_{ses}_task-rest_echo-{e}_bold.NORDIC_off.ThermalNoise.txt')
        aux      = np.loadtxt(aux_path)
        df_list.append( {'Subject':sbj,'Session':ses,'m-NORDIC':'off','Echo':e,'Thermal Noise':np.mean(aux[10::])})
        aux_path = osp.join(PRCS_DATA_DIR,sbj,'D02_NORDIC',f'{sbj}_{ses}_task-rest_echo-{e}_bold.NORDIC_on.ThermalNoise.txt')
        aux      = np.loadtxt(aux_path)
        df_list.append( {'Subject':sbj,'Session':ses,'m-NORDIC':'on','Echo':e,'Thermal Noise':np.mean(aux[10::])})
df = pd.DataFrame(df_list)


# Select the scans from session 1 with the largest change in thermal noise before and after applying m-NORDIC

# In[4]:


aux = (100 * (df.set_index(['Subject','Session','m-NORDIC','Echo']).loc[:,:,'off',:] - df.set_index(['Subject','Session','m-NORDIC','Echo']).loc[:,:,'on',:]) 
       / df.set_index(['Subject','Session','m-NORDIC','Echo']).loc[:,:,'off',:])
aux.loc[:,'ses-1',2].sort_values(by='Thermal Noise',ascending=False).iloc[0:2].round(2)


# ***
# 
# # Create AFNI Snapshots
# 
# For the two scans of interest, we must:
# 
# 1) Compute the diff between NORDIC on and NORDIC off. We do this for no regression and basic regression.
# 2) Compute the mean across time of these Diff maps
# 3) Compute the stdv across time of these Diff maps
# 4) Run a seed-based correlation (seed in visual cortex) on both Diff maps.
# 5) Take snapshots of different views of these results in AFNI
# 
# The following scripts will perform these operations in the two scans listed on the cell above
# 
# ```bash
# 
# cd ${BASH_DIR}
# export SBJ=sub-20 SES=ses-1; sh ./S10_SuppFigure04_GetIndividualPanels.sh
# export SBJ=sub-16 SES=ses-1; sh ./S10_SuppFigure04_GetIndividualPanels.sh
# ```

# ***
# # Generate Left Side of Figure

# In[5]:


left_panel_sbj = 'sub-20'
ses = 'ses-1'


# Grab all the relevant AFNI snapshots needed for the left most column of the figure.

# In[6]:


mean_ax_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_mean.axi.png'
mean_sg_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_mean.sag.png'
mean_cr_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_mean.cor.png'

stdv_ax_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_stdv.axi.png'
stdv_sg_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_stdv.sag.png'
stdv_cr_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_stdv.cor.png'

no_reg_sbco_ax_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_no_reg_sbco.axi.png'
no_reg_sbco_sg_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_no_reg_sbco.sag.png'
no_reg_sbco_cr_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_no_reg_sbco.cor.png'

basic_sbco_ax_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_basic_reg_sbco.axi.png'
basic_sbco_sg_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_basic_reg_sbco.sag.png'
basic_sbco_cr_path = f'./figures/pBOLD_SuppFig04_{left_panel_sbj}_{ses}_basic_reg_sbco.cor.png'

style_1 = {'font-family':'sans-serif','font-size':'22px', 'writing-mode':'vertical-lr', 'text-orientation':'sideways', 'transform':'rotate(180deg)','text-align':'center'}


# In[7]:


left_top_header = pn.pane.HTML("<div style='font-family: sans-serif; font-size: 22px;'>Example Scan 1 | 38% Thermal Noise Reduction</div>", width=700, styles={'text-align': 'center'})
panel_id_a      = pn.pane.HTML("<div>(a) Mean</div>",          width=15, height=200, styles=style_1)
panel_id_b      = pn.pane.HTML("<div>(b) St. Deviation</div>", width=15, height=200, styles=style_1)
panel_id_c      = pn.pane.HTML("<div>(c) Seed Based R</div>", width=15, height=200, styles=style_1)
panel_id_d      = pn.pane.HTML("<div>(d) Seed Based R</div>", width=15, height=200, styles=style_1)

a_left = pn.Row(panel_id_a, pn.pane.PNG(mean_ax_path, height=200),pn.pane.PNG(mean_sg_path, height=200),pn.pane.PNG(mean_cr_path, height=200))
b_left = pn.Row(panel_id_b,pn.pane.PNG(stdv_ax_path, height=200),pn.pane.PNG(stdv_sg_path, height=200),pn.pane.PNG(stdv_cr_path, height=200))
c_left = pn.Row(panel_id_c,pn.pane.PNG(no_reg_sbco_ax_path, height=200),pn.pane.PNG(no_reg_sbco_sg_path, height=200),pn.pane.PNG(no_reg_sbco_cr_path, height=200))
d_left = pn.Row(panel_id_d,pn.pane.PNG(basic_sbco_ax_path, height=200),pn.pane.PNG(basic_sbco_sg_path, height=200),pn.pane.PNG(basic_sbco_cr_path, height=200))


# ***
# # Generate Right Side of Figure

# In[8]:


right_panel_sbj = 'sub-16'
ses = 'ses-1'


# In[9]:


mean_ax_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_mean.axi.png'
mean_sg_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_mean.sag.png'
mean_cr_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_mean.cor.png'

stdv_ax_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_stdv.axi.png'
stdv_sg_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_stdv.sag.png'
stdv_cr_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_stdv.cor.png'

no_reg_sbco_ax_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_no_reg_sbco.axi.png'
no_reg_sbco_sg_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_no_reg_sbco.sag.png'
no_reg_sbco_cr_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_no_reg_sbco.cor.png'

basic_sbco_ax_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_basic_reg_sbco.axi.png'
basic_sbco_sg_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_basic_reg_sbco.sag.png'
basic_sbco_cr_path = f'./figures/pBOLD_SuppFig04_{right_panel_sbj}_{ses}_basic_reg_sbco.cor.png'


# In[10]:


right_top_header = pn.pane.HTML("<div style='font-family: sans-serif; font-size: 22px;'>Example Scan 2 | 34% Thermal Noise Reduction</div>", width=700, styles={'text-align': 'center'})

a_right = pn.Row(pn.pane.PNG(mean_ax_path, height=200),pn.pane.PNG(mean_sg_path, height=200),pn.pane.PNG(mean_cr_path, height=200))
b_right = pn.Row(pn.pane.PNG(stdv_ax_path, height=200),pn.pane.PNG(stdv_sg_path, height=200),pn.pane.PNG(stdv_cr_path, height=200))
c_right = pn.Row(pn.pane.PNG(no_reg_sbco_ax_path, height=200),pn.pane.PNG(no_reg_sbco_sg_path, height=200),pn.pane.PNG(no_reg_sbco_cr_path, height=200))
d_right = pn.Row(pn.pane.PNG(basic_sbco_ax_path, height=200),pn.pane.PNG(basic_sbco_sg_path, height=200),pn.pane.PNG(basic_sbco_cr_path, height=200))


# ***
# # Combine all elements

# In[12]:


pn.Row(pn.Column(left_top_header,a_left,b_left,c_left, d_left),
pn.Column(right_top_header,a_right,b_right,c_right, d_right)).save('./figures/pBOLD_SuppFig04.html')


# Here is the figure
# 
# ![Supp Figure 04](figures/pBOLD_SuppFig04.png)
