import hvplot.pandas
import holoviews as hv

from statannotations.Annotator import Annotator
from sfim_lib.plotting.fc_matrices import hvplot_fc
import panel as pn
import pandas as pd
import numpy as np
import xarray as xr
from nilearn.connectome import sym_matrix_to_vec
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

from .basics import TES_MSEC, echo_pairs

# Scatter Plot related functions
# ==============================
def gen_scatter(dataset,data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_linear_fit=False, ax_lim=None, hexbin=False):
    """
    Generate scatter plot for two different FC matrices

    Inputs:
    -------
    dataset (str): name of the dataset being used (e.g., evaluation, discovery)
    data_fc (dict): dictionary with FC matrices
    sbj (str): subject ID
    ses (str): session ID
    pp (str): pre-processing pipeline
    nordic (str): whether Nordic was used or not
    eep1 (str): first echo pair in the format 'e02|e02'
    eep2 (str): second echo pair in the format 'e02|e02'
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    show_linear_fit (bool): whether to show linear fit line or not
    ax_lim (float): axis limits for the scatter plot
    hexbin (bool): whether to use hexbin plot instead of scatter plot

    Returns:
    --------
    hvplot object with the scatter plot and theoretical lines

    """
    echo_times_dict = TES_MSEC[dataset]

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

def fc_across_echoes_scatter_page(dataset,fc_data,qa_data,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, show_line=False, ax_lim=None, other_stats=None, hexbin=False):
    """
    Create panel Frame with scatter plots for all FC combinations and table with QC metrics

    Inputs:
    -------
    dataset (str): name of the dataset being used (e.g., evaluation, discovery)
    fc_data (dict): dictionary with FC matrices
    qa_data (xarray.DataArray): xarray object with quality metrics
    sbj (str): subject ID
    ses (str): session ID
    pp (str): pre-processing pipeline
    nordic (str): whether Nordic was used or not
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    pairs_of_echo_pairs (list): list of echo pair combinations in the format ['e02|e02_vs_e03|e03', ...]
    show_line (bool): whether to show linear fit line or not
    ax_lim (float): axis limits for the scatter plot
    other_stats (pd.DataFrame or None): additional statistics to be displayed in the table
    hexbin (bool): whether to use hexbin plot instead of scatter plot

    Returns:
    --------
    pn.Row: Panel layout containing the scatter plots and statistics table
    """

    # Grid of scatter plots
    scatter_layout = pn.layout.GridBox(ncols=5)
    for i in pairs_of_echo_pairs:
        eep1,eep2=i.split('_vs_')
        plot = gen_scatter(dataset,fc_data,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_line, ax_lim, hexbin=hexbin)
        scatter_layout.append(plot)

    # Statistics Table 
    stats_df      = qa_data.loc[sbj,ses,pp,nordic,fc_metric,:,:].to_dataframe(name='QC').reset_index().drop(['sbj','ses','pp','fc_metric'],axis=1).pivot(index='ee_vs_ee', columns='qc_metric', values='QC')
    stats_mean_df = pd.DataFrame(stats_df.mean(),columns=['Avg']).T
    if other_stats is None:
        tables = pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2))   )
    else:
        tables = pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2)), pn.layout.Divider(),  pn.pane.DataFrame(other_stats))

    # Create Page
    frame = pn.Row(scatter_layout, tables)
    return frame
    

# FC Matrix Plotting functions
# ============================
def get_fc_matrix(data, qa_xr, sbj, ses, pp, nordic, fc_metric, echo_pair='e02|e02', net_cmap='viridis', ax_lim=None, title=None):
    """
    Get an hvplot-based FC matrix

    Inputs:
    -------
    data (dict): dictionary with FC matrices
    qa_xr (xarray.DataArray): xarray object with quality metrics
    sbj (str): subject ID
    ses (str): session ID
    pp (str): pre-processing pipeline
    nordic (str): whether Nordic was used or not
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    echo_pair (str): echo pair in the format 'e02|e02'
    net_cmap (str): colormap for networks
    ax_lim (float or None): axis limits for the colorbar, if None, default limits are used
    title (str or None): title for the plot, if None, a default title is generated

    Returns:
    --------
    hvplot object with the FC matrix plot
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
    
    Inputs:
    -------
    data (dict): dictionary with FC matrices
    qa_xr (xarray.DataArray): xarray object with quality metrics
    sbj (str): subject ID
    ses (str): session ID
    nordic (str): whether Nordic was used or not
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    echo_pair (str): echo pair in the format 'e02|e02'
    net_cmap (str): colormap for networks
    title (str or None): title for the plot, if None, a default title is generated

    Returns:
    --------
    pn.Row: Panel layout containing the FC matrices for different pipelines
    """
    if fc_metric == 'R':
        cbar_title = "Pearson's Correlation:"
        clim = (-.8,.8)
    if fc_metric == 'C':
        cbar_title = "Covariance:"
        clim = (-.5,.5)
    layout = pn.Row()
    for pp, pp_label in zip(['ALL_Basic','ALL_GS','ALL_Tedana-fastica','ALL_Tedana-robustica'],
                            ['Basic Regression','Global Signal Regression','Tedana Denoising (fastica)','Tedana Denoising (robustica)']):
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
def get_static_report(qa,fc_metric,qc_metric,x='Pre-processing',hue='NORDIC',show_stats=False, stat_test='t-test_paired',stat_annot_type='star', legend_location='best', remove_outliers_from_swarm=True, palette='Set2', show_points=False,session='all'):
    """
    Create Static Bar Graph for a given quality metric

    Inputs:
    -------
    qa (xarray.DataArray or pd.DataFrame): quality assessment data
    fc_metric (str): functional connectivity metric to be used
    qc_metric (str): quality control metric to be plotted
    x (str): column name for the x-axis (default: 'Pre-processing')
    hue (str): column name for the hue (default: 'NORDIC')
    show_stats (bool): whether to show statistical annotations or not
    stat_test (str): statistical test to be used for annotations
    stat_annot_type (str): type of annotation to be used ('star' or 'text')
    legend_location (str): location of the legend in the plot
    remove_outliers_from_swarm (bool): whether to remove outliers from swarm plot
    palette (str): color palette to be used for the plot
    show_points (bool): whether to show individual data points in the plot
    session (str): session filter

    Returns:
    --------
    matplotlib figure: bar plot with quality metrics
    """
    if isinstance(qa, xr.DataArray):
        df         = qa.mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
        df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
        df         = df.replace({'ALL_Basic':'Basic','ALL_GS':'GSR','ALL_Tedana-fastica':'Tedana-fastica','ALL_Tedana-robustica':'Tedana-robustica', 'NORDIC':'On'})
    elif isinstance(qa, pd.DataFrame):
        df = qa.copy()
    else:
        print("++ ERROR [get_barplot]: Expected a xarray or pandas dataframe but got something else. Function exiting")
        return None
    if session !='all':
        df = df[df['Session']==session]
    # Extract available values for X and HUE
    x_options   = list(df[x].unique())
    hue_options = list(df[hue].unique())
    num_x_categories = len(x_options)
    num_hue_categories = len(hue_options)
    
    df_swarm = df.copy()
    if remove_outliers_from_swarm:
        quantile_value = df[qc_metric].quantile(.97)
        df_swarm[qc_metric]=df_swarm[qc_metric].where(df_swarm[qc_metric] <= quantile_value, np.nan)
        df_swarm.dropna(inplace=True)
    pairs  = [((x,h[1]),(x,h[0])) for x in x_options for h in combinations(hue_options,2)]
    colors = sns.color_palette(palette,num_hue_categories) 

    sns.set_context("paper", rc={"xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 16, 'legend.fontsize':16})
    fig, axs = plt.subplots(1,1,figsize=(6,6));
    sns.despine(top=True, right=True)
    sns.barplot(data=df,hue=hue, y=qc_metric, x=x, alpha=0.5, ax =axs, errorbar=('ci',95), palette=colors);
    if show_points:
        sns.swarmplot(data=df_swarm,hue=hue, y=qc_metric, x=x, ax =axs, s=1, dodge=True, legend=False, palette=colors);
    
    if show_stats:
        annotation = Annotator(axs, pairs, data=df, x=x, y=qc_metric, hue=hue);
        annotation.configure(test=stat_test, text_format=stat_annot_type, loc='inside', verbose=0);
        annotation.apply_test(alternative='two-sided');
        annotation.annotate();
    sns.move_legend(axs, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,)
    plt.tight_layout()
    plt.close()
    return fig
    

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