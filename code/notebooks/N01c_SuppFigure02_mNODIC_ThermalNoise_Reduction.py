#!/usr/bin/env python
# coding: utf-8

# import panel as pn
# pn.extension()

# # Description: Generate Supplementary Figure 02
# 
# This notebook takes the outputs from N01a and N02b, adds headers and creates the final version of Supp. Figure 02.
# 
# ## Create Top Banners

# In[36]:


top_banner_a = pn.pane.HTML(
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
      (a) Discovery Dataset
    </div>
    """,
    sizing_mode="scale_width",
)

top_banner_b = pn.pane.HTML(
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
      (b) Evaluation Dataset
    </div>
    """,
    sizing_mode="scale_width",
)


# ## Compose Final Figure Layout

# In[38]:


pn.Row(pn.Column(top_banner_a,pn.pane.PNG('./figures/pBOLD_SuppFig02_a.png',width=1200)),
       pn.Column(top_banner_b,pn.pane.PNG('./figures/pBOLD_SuppFig02_b.png',width=1200)), width=2410).save('./figures/pBOLD_SuppFig02.html')


# Static version of the figure
# 
# ![Suppl. Figure 02](figures/pBOLD_SuppFig02.png)
