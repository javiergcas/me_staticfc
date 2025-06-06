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
from itertools import combinations

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
        layout.append(fc_plot)
    return layout

# Covariance Carpet Plot Functions:
# ==================================
def get_cov_heatmap(data,sbj,ses,pp_opts,nordic_opts,roi_info=None,clim=5, echo_pairs=echo_pairs):
    norid_cards = {'Off':pn.Card(title='Nordic Off'), 'On':pn.Card(title='Nordic On')}
    for nordic in nordic_opts.values():
        for pp in pp_opts.values():
            df = pd.DataFrame([data[(sbj, ses, pp, nordic, ep, 'C')] for ep in echo_pairs], index=echo_pairs)
            if roi_info is not None:
                df.columns = ['|'.join(row) for row in roi_info[['ROI_Name','Hemisphere','Network']].values]
            norid_cards[nordic].append(df.hvplot.heatmap(width=2000, height=150, hover_cols=['Network'], title=pp, clim=(0,clim), cmap='viridis').opts(xrotation=90, xaxis=None))
    layout = pn.Column(norid_cards['On'],pn.layout.Divider(), norid_cards['Off'])
    return layout

# Regional Co-variance Scatter Plots
# ==================================
def gen_roi_cov_scatter(data,sbj,ses,pp,nordic, eep1,eep2, show_linear_fit=False, ax_lim=None, roi_info=None, cmap=None):
    """
    Generate scatter plot for two different FC matrices
    """
    if (sbj,ses,pp,nordic,eep1,'C') not in data:
        return pn.pane.Markdown('#Not Available')
    data_df = pd.DataFrame(columns=[eep1,eep2])
    if roi_info is None:
        data_df = pd.DataFrame(columns=[eep1,eep2])
    else:
        data_df = pd.DataFrame(columns=[eep1,eep2], index=roi_info.set_index(['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']).index)            
    data_df[eep1] = data[sbj,ses,pp,nordic,eep1,'C']
    data_df[eep2] = data[sbj,ses,pp,nordic,eep2,'C']
    # Compute limits for X and Y axis
    if ax_lim is None:
        lims = (data_df.quantile(0.01).min(),data_df.quantile(0.99).max())
    else:
        lims=(-ax_lim,ax_lim)
    # Create scatter plot and fitted line
    if (roi_info is not None) and (cmap is not None):
        data_df['NW_Color'] = [cmap[c] for c in data_df.index.get_level_values('Network')]
        scat           = data_df.hvplot.scatter(x=eep1, y=eep2, aspect='square',s=5, xlim=lims, ylim=lims, alpha=.7, hover_cols=['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network'], color='NW_Color') #.opts(active_tools=['save'], tools=['save'])
    else:
        scat           = data_df.hvplot.scatter(x=eep1, y=eep2, aspect='square',s=5, xlim=lims, ylim=lims, alpha=.7, hover_cols=['ROI_Name', 'ROI_ID', 'Hemisphere', 'Network']) #.opts(active_tools=['save'], tools=['save'])
    data_lin_fit   = hv.Slope.from_scatter(scat).opts(line_width=3, line_color='#0f0fff') #.opts(active_tools=['save'], tools=['save'])

    # Compute theoretical slopes for extreme BOLD and So dominated regimes
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

def cov_across_echoes_scatter_page(cov_data,qa_data,sbj,ses,pp, nordic, pairs_of_echo_pairs, show_line=False, ax_lim=None, other_stats=None, roi_info=None, cmap=None):
    """
    Create Frame with scatter plots for all FC combinations and table with QC metrics
    """
    # Grid of scatter plots
    scatter_layout = pn.layout.GridBox(ncols=5)
    for i in pairs_of_echo_pairs:
        eep1,eep2=i.split('_vs_')
        plot = gen_roi_cov_scatter(cov_data,sbj,ses,pp,nordic,eep1,eep2, show_line, ax_lim, roi_info=roi_info, cmap=cmap)
        scatter_layout.append(plot)

    # Statistics Table 
    stats_df      = qa_data.loc[sbj,ses,pp,nordic,'C',:,:].to_dataframe(name='QC').reset_index().drop(['sbj','ses','pp','fc_metric'],axis=1).pivot(index='ee_vs_ee', columns='qc_metric', values='QC')
    stats_mean_df = pd.DataFrame(stats_df.mean(),columns=['Avg']).T
    if other_stats is None:
        tables = pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2))   )
    else:
        tables = pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2)), pn.layout.Divider(),  pn.pane.DataFrame(other_stats.loc[(sbj,ses)]) )

    # Create Page
    frame = pn.Row(scatter_layout, tables)
    return frame
    
# Static Group Level Report Functions
# ====================================
def get_barplot(qa_xr,nordic,fc_metric,qc_metric,x='Pre-processing',hue='NORDIC',show_stats=False, stat_test='t-test_paired',stat_annot_type='star', legend_location='best', remove_outliers_from_swarm=True):
    """
    Create Static Bar Graph for a given quality metric
    """
    df         = qa_xr.mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
    df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
    df         = df.replace({'ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana','ALL_Tedana-NORDIC_FixNComps':'Tedana (n=88)', 'NORDIC':'On'})
    num_hues   = len(list(df[hue].unique()))
    df_swarm = df.copy()
    if remove_outliers_from_swarm:
        quantile_value = df[qc_metric].quantile(.97)
        df_swarm[qc_metric]=df_swarm[qc_metric].where(df_swarm[qc_metric] <= quantile_value, np.nan)

    if (x=='Pre-processing') and (hue=='NORDIC'):
        pairs  = [((p,'On'),(p,'Off')) for p in df[x].unique()]
        colors = sns.color_palette("rocket",num_hues)
    if (x=='NORDIC') and (hue=='Pre-processing'):
        pairs      = [(('On',c[0]),('On',c[1])) for c in combinations(list(df['Pre-processing'].unique()),2)]
        pairs      = pairs + [(('Off',c[0]),('Off',c[1])) for c in combinations(list(df['Pre-processing'].unique()),2)]
        colors = sns.color_palette("Set2",num_hues)

    sns.set_context("paper", rc={"xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 16, 'legend.fontsize':16})
    fig, axs = plt.subplots(1,1,figsize=(6,6));
    sns.despine(top=True, right=True)
    sns.barplot(data=df,hue=hue, y=qc_metric, x=x, alpha=0.5, ax =axs, errorbar=('ci',95), palette=colors);
    sns.swarmplot(data=df_swarm,hue=hue, y=qc_metric, x=x, ax =axs, s=.5, dodge=True, legend=False, palette=colors);
    
    if show_stats:
        annotation = Annotator(axs, pairs, data=df, x=x, y=qc_metric, hue=hue);
        annotation.configure(test=stat_test, text_format=stat_annot_type, loc='inside', verbose=0);
        annotation.apply_test(alternative='two-sided');
        annotation.annotate();
    sns.move_legend(axs, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,)
    plt.tight_layout()
    plt.close()
    return fig

def get_barplot_discovery_dataset(qa_xr,nordic,fc_metric,qc_metric,x='Pre-processing',hue='Session',show_stats=False, stat_test='t-test_paired',stat_annot_type='star', legend_location='best'):
    """
    Create Static Bar Graph for a given quality metric
    """
    df         = qa_xr.sel(fc_metric=fc_metric, nordic=nordic, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric','nordic'],axis=1).reset_index()
    df.columns = ['Subject','Session','Pre-processing',qc_metric]
    df         = df.replace({'constant_gated':'Constant TR','cardiac_gated':'Cardiac Gating','ALL_Basic':'Basic','ALL_GSasis':'GSR','ALL_Tedana':'Tedana','ALL_Tedana-NORDIC_FixNComps':'Tedana (n=88)', 'NORDIC':'On'})
    num_hues   = len(list(df[hue].unique()))
    
    if (x=='Pre-processing') and (hue=='Session'):
        pairs  = [((p,'Constant TR'),(p,'Cardiac Gating')) for p in df[x].unique()]
        colors = sns.color_palette("rocket",num_hues)
    if (x=='Session') and (hue=='Pre-processing'):
        pairs      = [(('Constant TR',c[0]),('Constant TR',c[1])) for c in combinations(list(df[hue].unique()),2)]
        pairs      = pairs + [(('Cardiac Gating',c[0]),('Cardiac Gating',c[1])) for c in combinations(list(df[hue].unique()),2)]
        colors = sns.color_palette("Set2",num_hues)
        
    fig, axs = plt.subplots(1,1,figsize=(6,6));
    sns.barplot(data=df,hue=hue, y=qc_metric, x=x, alpha=0.5, ax =axs, errorbar=('ci',95), palette=colors);
    sns.swarmplot(data=df,hue=hue, y=qc_metric, x=x, ax =axs, s=.5, dodge=True, legend=False, palette=colors);
    
    if show_stats:
        annotation = Annotator(axs, pairs, data=df, x=x, y=qc_metric, hue=hue);
        annotation.configure(test=stat_test, text_format=stat_annot_type, loc='inside', verbose=0);
        annotation.apply_test(alternative='two-sided');
        annotation.annotate();
    sns.move_legend(axs, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,)
    plt.close()
    return fig
    
#def get_barplot(qa_xr,nordic, fc_metric,qc_metric):
#    
#    df= qa_xr.sel(fc_metric=fc_metric, nordic=nordic, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric','nordic'],axis=1).reset_index()
#    df.columns=['Subject','Data Type','Pre-processing',qc_metric]
#    df = df.replace({'constant_gated':'Constant TR','cardiac_gated':'Cardiac Gating', 'ALL_Basic':'Basic Regressors','ALL_GSasis':'GRS','ALL_Tedana':'Tedana'})
#
#    g = sns.catplot(data=df,kind='bar',x='Data Type',hue='Pre-processing',y=qc_metric, errorbar=('ci', 95), alpha=0.5)
#    sns.swarmplot(data=df, x="Data Type",hue='Pre-processing', y=qc_metric, size=1, dodge=True, legend=False)
#    g.set_axis_labels("", qc_metric)
#    g.despine(left=True)
    
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