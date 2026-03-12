#!/usr/bin/env python
# coding: utf-8

# # Description: Generate Suppl. Figure 3
# 
# Supplementary Figure 3 shows how TSNR and pBOLD change within each pre-processing pipeline depending on whether m-NORDIC is on or off.
# 
# All graphic elements needed for this figure are generated in the following notebooks:
# 
# * Panels a and b: ```N10b_TSNR_and_pBOLD_BarPlots_SuppFig03.discovery.ipnb```
# * Panels c and d: ```N10a_TSNR_and_pBOLD_BarPlots_Figure04.evaluation.ipnb```
# 

# In[4]:


import panel as pn
pn.extension()


# ## Create Top Banners

# In[11]:


top_banner_ab = pn.pane.HTML(
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
        width:1200px;
    ">
      Discovery Dataset
    </div>
    """,
    sizing_mode="scale_width",
)

top_banner_cd = pn.pane.HTML(
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
        width:1200px;
    ">
      Evaluation Dataset
    </div>
    """,
    sizing_mode="scale_width",
)


# ## Generate Final Figure Layout

# In[14]:


pn.Row(pn.Column(top_banner_ab,
                 pn.Row(pn.pane.PNG('./figures/pBOLD_SuppFig03_a.png',width=600),pn.pane.PNG('./figures/pBOLD_SuppFig03_b.png',width=600))),
       pn.Column(top_banner_cd,
                 pn.Row(pn.pane.PNG('./figures/pBOLD_SuppFig03_c.png',width=600),pn.pane.PNG('./figures/pBOLD_SuppFig03_d.png',width=600))), width=2410).save('./figures/pBOLD_SuppFig03.html')


# Here is the static version of the figure
# 
# ![Supp Figure 03](figures/pBOLD_SuppFig03.png)
