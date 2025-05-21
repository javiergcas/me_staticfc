import hvplot.pandas
import holoviews as hv

from statannotations.Annotator import Annotator
from sfim_lib.plotting.fc_matrices import hvplot_fc
import panel as pn
import pandas as pd
import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from .basics import TES_MSEC, SESSIONS, echo_pairs_tuples, echo_pairs, pairs_of_echo_pairs
echo_times_dict = TES_MSEC['Spreng_Scanner1']
ses_list        = SESSIONS['Spreng_Scanner1']

# Scatter Plot related functions
# ==============================
def gen_scatter(data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_linear_fit=False, ax_lim=None, hexbin=False):
    """
    Generate scatter plot for two different FC matrices
    """
    if (sbj,ses,pp,nordic,eep1,fc_metric) not in data_fc:
        return pn.pane.Markdown('#Not Available')
    data_df = pd.DataFrame(columns=[eep1,eep2, fc_metric])
    data_df[eep1] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep1,fc_metric].values, discard_diagonal=True)
    data_df[eep2] = sym_matrix_to_vec(data_fc[sbj,ses,pp,nordic,eep2,fc_metric].values, discard_diagonal=True)
    # Compute limits for X and Y axis
    if ax_lim is None:
        if fc_metric == 'R':
            lims = (-1,1) 
        else:
            lims = (data_df.quantile(0.01).min(),data_df.quantile(0.99).max())
    else:
        lims=(-ax_lim,ax_lim)
    # Create scatter plot and fitted line
    scat           = data_df.hvplot.scatter(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7) #.opts(active_tools=['save'], tools=['save'])
    data_lin_fit   = hv.Slope.from_scatter(scat).opts(line_width=3, line_color='#0f0fff') #.opts(active_tools=['save'], tools=['save'])

    if hexbin:
        scat = data_df.hvplot.hexbin(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7)
    # Compute theoretical slopes for extreme BOLD and So dominated regimes
    if fc_metric  == 'R':
        BOLD_line_sl, BOLD_line_int = 1.,0.
    else:
        e1_X,e2_X     = eep1.split('|')
        e1_Y,e2_Y     = eep2.split('|')
        BOLD_line_sl  = (echo_times_dict[e1_Y]*echo_times_dict[e2_Y])/(echo_times_dict[e1_X]*echo_times_dict[e2_X])
        BOLD_line_int = 0.
    
    So_line_sl, So_line_int = 1.,0.
    
    # Create Theoretical BOLD and So lines
    BOLD_line  = hv.Slope(BOLD_line_sl, BOLD_line_int).opts(line_color='g',line_width=2, line_dash='dashed') #.opts(active_tools=['save'], tools=['save'])
    So_line    = hv.Slope(So_line_sl,   So_line_int  ).opts(line_color='r',line_width=2, line_dash='dashed') #.opts(active_tools=['save'], tools=['save'])
    
    # Join all graphical elements
    if show_linear_fit:
        plot = (scat * data_lin_fit * So_line * BOLD_line) #.opts(toolbar=None)
    else:
        plot = (scat * So_line * BOLD_line)
     
    return plot

def fc_across_echoes_scatter_page(fc_data,qa_data,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=False, ax_lim=None, other_stats=None, hexbin=False):
    """
    Create Frame with scatter plots for all FC combinations and table with QC metrics
    """
    # Grid of scatter plots
    scatter_layout = pn.layout.GridBox(ncols=5)
    for i in pairs_of_echo_pairs:
        eep1,eep2=i.split('_vs_')
        plot = gen_scatter(fc_data,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_line, ax_lim, hexbin=hexbin)
        scatter_layout.append(plot)

    # Statistics Table 
    stats_df      = qa_data.loc[sbj,ses,pp,nordic,fc_metric,:,:].to_dataframe(name='QC').reset_index().drop(['sbj','ses','pp','fc_metric'],axis=1).pivot(index='ee_vs_ee', columns='qc_metric', values='QC')
    stats_mean_df = pd.DataFrame(stats_df.mean(),columns=['Avg']).T
    if other_stats is None:
        tables = pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2))   )
    else:
        tables = pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2)), pn.layout.Divider(),  pn.pane.DataFrame(other_stats.loc[(sbj,ses)]) )

    # Create Page
    frame = pn.Row(scatter_layout, tables)
    return frame
    

# FC Matrix Plotting functions
# ============================
def get_fc_matrix(data, qa_xr, sbj, ses, pp, nordic, fc_metric, echo_pair='e02|e02', net_cmap='viridis', ax_lim=None, title=None):
    """
    Get an hvplot-based FC matrix
    """
    # Set Title in colorbar
    if fc_metric == 'R':
        cbar_title = "Pearson's Correlation:"
    elif fc_metric == 'C':
        cbar_title = "Covariance:"
    # Set Color Limits
    if ax_lim is None:
        if fc_metric == 'R':
            clim = (-.8,.8)
        if fc_metric == 'C':
            clim = (-.5,.5)
    else:
        clim = (-ax_lim, ax_lim)
        
    # Gather FC for this particular set of parameters (e.g., sbj, ses, pp, etc...)
    fc_matrix = data[sbj,ses,pp,nordic,echo_pair,fc_metric]
    # Gather pBOLD for this particular set of parameters
    fc_pBOLD  = qa_xr.sel(sbj=sbj,ses=ses,pp=pp,qc_metric='pBOLD',fc_metric=fc_metric).mean().values

    if title is None:
        title = '%s | %s | p_BOLD (%s) = %.2f' %(sbj, pp, fc_metric, fc_pBOLD)
        
    fc_plot   = hvplot_fc(fc_matrix, 
                          major_label_overrides='regular_grid', net_cmap=net_cmap,
                          cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', clim=clim, cbar_title=cbar_title,
                          cbar_title_fontsize=14,ticks_font_size=14).opts(default_tools=["pan"]).opts(title=title)
    return fc_plot
    
def get_fc_matrices(data,qa_xr,sbj,ses,nordic, fc_metric, echo_pair='e02|e02', net_cmap='viridis', title=None):
    """
    Create a layout with FC matrices (R and C) for the different pipelines
    """
    if fc_metric == 'R':
        cbar_title = "Pearson's Correlation:"
        clim = (-.8,.8)
    if fc_metric == 'C':
        cbar_title = "Covariance:"
        clim = (-.5,.5)
    layout = pn.Row()
    for pp, pp_label in zip(['ALL_Basic','ALL_GSasis','ALL_Tedana'],['Basic Regression','Global Signal Regression','Tedana Denoising']):
        if (sbj,ses,pp,nordic,echo_pair,fc_metric) not in data:
            layout.append(pn.pane.Markdown('#Not Available'))
            continue

        fc_plot = get_fc_matrix(data, qa_xr, sbj, ses, pp, nordic, fc_metric, echo_pair='e02|e02', net_cmap=net_cmap, ax_lim=None, title=None)
        #fc_matrix = data[sbj,ses,pp,nordic,echo_pair,fc_metric]
        #fc_pBOLD  = qa_xr.sel(sbj=sbj,ses=ses,pp=pp,nordic='Off',qc_metric='pBOLD',fc_metric=fc_metric).mean().values
        #fc_plot   = hvplot_fc(fc_matrix, major_label_overrides='regular_grid', net_cmap=net_cmap,
        #                  cmap='RdBu_r', by='Network', add_labels=False, colorbar_position='left', clim=clim,
        #                  cbar_title=cbar_title,cbar_title_fontsize=14,ticks_font_size=14).opts(default_tools=["pan"]).opts(title='%s | p_BOLD (%s) = %.2f' %(pp_label,fc_metric,fc_pBOLD))
        layout.append(fc_plot)
    return layout

# Static Group Level Report Functions
# ====================================
# def get_barplot(qa_xr,fc_metric,qc_metric,x='Pre-processing',hue='NORDIC',stat_test='t-test_paired'):
#     """
#     Create Static Bar Graph for a given quality metric
#     """
#     df         = qa_xr.sel(fc_metric=fc_metric, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
#     df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
#     df         = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana','NORDIC':'On (Nc auto)','NORDIC_FixNComps':'On (Nc=88)'})

#     if (x=='Pre-processing') and (hue=='NORDIC'):
#         pairs      = [(("Basic","Off"),("Basic","On (Nc auto)")),(("Basic","On (Nc auto)"),("Basic","On (Nc=88)")),(("Basic","Off"),("Basic","On (Nc=88)")),
#                       (("GSR","Off"),("GSR","On (Nc auto)")),(("GSR","On (Nc auto)"),("GSR","On (Nc=88)")),(("GSR","Off"),("GSR","On (Nc=88)")),
#                       (("Tedana","Off"),("Tedana","On (Nc auto)")),(("Tedana","On (Nc auto)"),("Tedana","On (Nc=88)")),(("Tedana","Off"),("Tedana","On (Nc=88)"))]
#         colors = sns.color_palette("rocket",3)
#     if (x=='NORDIC') and (hue=='Pre-processing'):
#         pairs      = [(("Off","Basic"),("Off","GSR")),(("Off","GSR"),("Off","Tedana")),(("Off","Basic"),("Off","Tedana")),
#                      (("On (Nc auto)","Basic"),("On (Nc auto)","GSR")),(("On (Nc auto)","GSR"),("On (Nc auto)","Tedana")),(("On (Nc auto)","Basic"),("On (Nc auto)","Tedana")),
#                      (("On (Nc=88)","Basic"),("On (Nc=88)","GSR")),(("On (Nc=88)","GSR"),("On (Nc=88)","Tedana")),(("On (Nc=88)","Basic"),("On (Nc=88)","Tedana"))]
#         colors = sns.color_palette("Set2",3)
        
#     fig, axs = plt.subplots(1,1,figsize=(4,4))
#     sns.barplot(data=df,hue=hue, y=qc_metric, x=x, alpha=0.5, ax =axs, errorbar=('ci',95), palette=colors)
#     sns.swarmplot(data=df,hue=hue, y=qc_metric, x=x, ax =axs, s=.5, dodge=True, legend=False, palette=colors)
    
#     annotation = Annotator(axs, pairs, data=df, x=x, y=qc_metric, hue=hue)
#     annotation.configure(test=stat_test, text_format='star', loc='inside', verbose=0)
#     annotation.apply_test(alternative='two-sided')
#     annotation.annotate()
#     return fig

def get_barplot(qa_xr,nordic, fc_metric,qc_metric):
    
    df= qa_xr.sel(fc_metric=fc_metric, nordic=nordic, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric','nordic'],axis=1).reset_index()
    df.columns=['Subject','Data Type','Pre-processing',qc_metric]
    df = df.replace({'constant_gated':'Constant TR','cardiac_gated':'Cardiac Gating', 'ALL_Basic':'Basic Regressors','ALL_GSasis':'GRS','ALL_Tedana':'Tedana'})

    g = sns.catplot(data=df,kind='bar',x='Data Type',hue='Pre-processing',y=qc_metric, errorbar=('ci', 95), alpha=0.5)
    sns.swarmplot(data=df, x="Data Type",hue='Pre-processing', y=qc_metric, size=3, dodge=True, legend=False)
    g.set_axis_labels("", qc_metric)
    g.despine(left=True)
    
    return pn.pane.Matplotlib(g.figure, tight=True)
    

# Dynamic Group Level Report Functions
# ====================================
def dynamic_summary_plot_gated(qa_xr, fc_metric, qc_metric, nordic):
    df= qa_xr.sel(fc_metric=fc_metric, nordic=nordic, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric','nordic'],axis=1).reset_index()
    df.columns=['Subject','Session','Pre-processing',qc_metric]
    df = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana', 'constant_gated':'Constant TR','cardiac_gated':'Cardiac Gated'})
    df['Scenario'] = df['Session']+'\n'+df['Pre-processing']

    plot = df.hvplot.box(   by='Scenario',y=qc_metric, legend=False) * \
           df.hvplot.scatter(x='Scenario',y=qc_metric, by='Subject', legend=False, hover_cols=['Subject','Session']) * \
           df.hvplot.line(by=['Subject','Session'],x='Scenario',y=qc_metric, legend=False, c='k', line_dash='dashed', line_width=0.5)
    return plot.opts(legend_position='top', height=400, width=600, legend_cols=4)

#def get_hv_box_discoveryset(qa_xr, qc_metric, fc_metric='C', NORDIC=False):
#    """
#    Create Dynamic Group Level Bar plots for a given quality metric
#
#    This only applies to the Discovery Set
#    """
#    if NORDIC:
#        pp=['ALL_Basic_NORDIC','ALL_GSasis_NORDIC','ALL_Tedana_NORDIC']
#        title='NORDIC ON'
#    else:
#        title='NORDIC OFF'
#        pp=['ALL_Basic','ALL_GSasis','ALL_Tedana']
#    df= qa_xr.sel(fc_metric=fc_metric, qc_metric=qc_metric, pp=pp).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
#    df.columns=['Subject','Session','Pre-processing',qc_metric]
#    df = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana', 'ALL_Basic_NORDIC':'Basic','ALL_GSasis_NORDIC':'GSR','ALL_Tedana_NORDIC':'Tedana'})
#    
#    df2=df.pivot(columns=['Pre-processing'],values=qc_metric, index=['Subject','Session'])
#    
#    plot_box     = df.hvplot.box(legend=False, by=['Session','Pre-processing'],box_color='Pre-processing',y=qc_metric).opts(cmap=['gray','lightgray','white'], width=500, height=500,ylabel=qc_metric, fontscale=1.5, title=title)
#    df['group'] = df['Session']+'|'+df['Pre-processing']
#    df = df.set_index('group').loc[['cardiac_gated|Basic','cardiac_gated|GSR','cardiac_gated|Tedana','constant_gated|Basic','constant_gated|GSR','constant_gated|Tedana']].reset_index()
#    plot_scatter = df.hvplot.scatter(legend=False, x='group',y=qc_metric,hover_cols=['Subject'],color=['Subject'],cmap=['red','green','blue','cyan','yellow','magenta','orange']).opts( width=500, height=500,ylabel=qc_metric, fontscale=1.5, #title=title, jitter=.03,xrotation=45) * \
#                   df.hvplot.line(by=['Subject','Session'],x='group',y=qc_metric, legend=False, c='k', line_dash='dashed', line_width=0.5)
#    return pn.Column(plot_box,plot_scatter) 
    
#def get_hv_box(qa_xr,qc_metric, fc_metric='C', qc_metric_selector='Basic<0.6'):
#    """
#    Create Dynamic Group Level Bar plots for a given quality metric
#    It also highlights a set of selected scans
#    """
#    #def update_heatmap(x, y):
#    #    new_data = pd.DataFrame(np.random.rand(10, 10), columns=range(10), index=range(10))
#    #    return hv.Image(new_data.values).opts(cmap='viridis', colorbar=True)
#    
#    df= qa_xr.sel(fc_metric=fc_metric, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
#    df.columns=['Subject','Session','Pre-processing',qc_metric]
#    df = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana', 'ALL_Basic_NORDIC':'Basic (NORDIC)','ALL_GSasis_NORDIC':'GSR (NORDIC)','ALL_Tedana_NORDIC':'Tedana (NORDIC)'})
#
#    df2=df.pivot(columns=['Pre-processing'],values=qc_metric, index=['Subject','Session'])
#    selected_scans = list(df2.query(qc_metric_selector).index)
#
#    plot_box     = df2.hvplot.box(legend=False).opts(box_color='Pre-processing', cmap=['lightblue','orange','green'], width=500, height=500,ylabel=qc_metric, fontscale=1.5)
#    plot_scatter = df.hvplot.scatter(y=qc_metric, x='Pre-processing', c='Pre-processing', legend=False).opts(size=1,jitter=0.05)
#    #tap_stream   = hv.streams.Tap(source=plot_scatter, x=np.nan, y=np.nan)
#
#    if len(selected_scans)>0:
#        plot = plot_box * plot_scatter * df.set_index(['Subject','Session']).loc[selected_scans].hvplot.line(by=['Subject','Session'],x='Pre-processing',y=qc_metric, legend=False, c='k', line_dash='dashed', line_width=0.5)
#    else:
#        plot = plot_box * plot_scatter
#    
#    #heatmap_dmap = hv.DynamicMap(lambda x, y: update_heatmap(x, y), streams=[tap_stream])
#    return plot #pn.Column(plot,heatmap_dmap)