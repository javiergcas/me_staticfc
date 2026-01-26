#!/usr/bin/env python
# coding: utf-8

# # Description: Evaluate NORDIC ability at removing Thermal Noise (Evaluation Dataset)
# 
# Prior to using this notebook, you need to run the ```bash/S03_ThermalNoiseEstimation...``` script to estimate thermal noise in all scans and TEs from both datasets.
# 
# For each scan, this script will write two files:
# 
# * ```{sbj}_{ses}_task-rest_echo-{e}_bold.NORDIC_off.ThermalNoise.txt```: holds thermal noise estimates before applying NORDIC.
# * ```{sbj}_{ses}_task-rest_echo-{e}_bold.NORDIC_on.ThermalNoise.txt```: holds thermal noise estimates after applying NORDIC.
# 
# Expectations:
# 
# * Thermal noise should be TE independent, yet becuase the suboptimal way in which we are measuring thermal noise, a slight difference migth occur.
# * Thermal noise should be significantly smaller after NORDIC for all echoes.

# In[1]:


import pandas as pd
import os.path as osp
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.basics import PRCS_DATA_DIR, PRJ_DIR
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt


# 
# ### Load list of scans from the Evaluation dataset entering the final set of analyses

# In[2]:


dataset_info_df = pd.read_csv(osp.join(PRJ_DIR,'resources','good_scans.txt'))
dataset_info_df = dataset_info_df.set_index(['Subject','Session'])
print('++ Number of scans: %s scans' % dataset_info_df.shape[0])


# ### Load the Thermmal noise estiamtes for these scans

# In[3]:


df_list = []
for sbj,ses in tqdm(dataset_info_df.index, desc='Scan'):
    for e in range(1,4):
        aux_path = osp.join(PRCS_DATA_DIR,sbj,'D02_NORDIC',f'{sbj}_{ses}_task-rest_echo-{e}_bold.NORDIC_off.ThermalNoise.txt')
        aux      = np.loadtxt(aux_path)
        df_list.append( {'Subject':sbj,'Session':ses,'NORDIC':'off','Echo':e,'Thermal Noise':np.mean(aux[10::])})
        aux_path = osp.join(PRCS_DATA_DIR,sbj,'D02_NORDIC',f'{sbj}_{ses}_task-rest_echo-{e}_bold.NORDIC_on.ThermalNoise.txt')
        aux      = np.loadtxt(aux_path)
        df_list.append( {'Subject':sbj,'Session':ses,'NORDIC':'on','Echo':e,'Thermal Noise':np.mean(aux[10::])})
df = pd.DataFrame(df_list)


# ### Separate the data by session ID

# In[4]:


df_ses1 = df.set_index('Session').loc['ses-1'].copy().reset_index(drop=True)
df_ses2  = df.set_index('Session').loc['ses-2'].copy().reset_index(drop=True)


# ### Print the overall improvement in Thermal Noise for the whole dataset

# In[5]:


aux = (100 * (df.set_index(['Subject','Session','NORDIC','Echo']).loc[:,:,'off',:] - df.set_index(['Subject','Session','NORDIC','Echo']).loc[:,:,'on',:]) / df.set_index(['Subject','Session','NORDIC','Echo']).loc[:,:,'off',:]).describe()
print('Thermal Noise reduction after NORDIC (%%): %0.2f +/- %0.2f' % (aux.loc['mean'].values[0],aux.loc['std'].values[0]))


# ### Generate Suppl. Figure 2.b panel

# In[6]:


fig, axs = plt.subplots(1,2,figsize=(14,7))
# Results for Constant Gated
sns.barplot(  data=df_ses1,hue='NORDIC',x='Echo',y='Thermal Noise', palette='Set2', errorbar='ci', ax=axs[0], alpha=0.5).set_title('Sesion 1')
sns.swarmplot(data=df_ses1,hue='NORDIC',x='Echo',y='Thermal Noise', palette='Set2',ax=axs[0], dodge=True, legend=False, size=1)
pairs=[((1,'off'),(1,'on')),((2,'off'),(2,'on')),((3,'off'),(3,'on'))]
annotation = Annotator(axs[0], pairs, data=df_ses1, hue='NORDIC', y='Thermal Noise',x='Echo')
annotation.configure(test='t-test_paired', loc='inside', verbose=0, comparisons_correction="Bonferroni");
annotation.apply_test(alternative='two-sided');
annotation.annotate();
# Results for Cardiac Gated
sns.barplot(  data=df_ses2,hue='NORDIC',x='Echo',y='Thermal Noise', palette='Set2', errorbar='ci', ax=axs[1], alpha=0.5).set_title('Sesion 2')
sns.swarmplot(data=df_ses2,hue='NORDIC',x='Echo',y='Thermal Noise', palette='Set2',ax=axs[1], dodge=True, legend=False, size=1)
pairs=[((1,'off'),(1,'on')),((2,'off'),(2,'on')),((3,'off'),(3,'on'))]
sns.despine()
annotation = Annotator(axs[1], pairs, data=df_ses2, hue='NORDIC', y='Thermal Noise',x='Echo')
annotation.configure(test='t-test_paired', loc='inside', verbose=0, comparisons_correction="Bonferroni");
annotation.apply_test(alternative='two-sided');
annotation.annotate();


# In[ ]:




