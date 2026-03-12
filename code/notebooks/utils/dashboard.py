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
from bokeh.models import DatetimeTickFormatter, DatetimeTicker


from .basics import TES_MSEC, echo_pairs, LABEL_MAPPING

# Scatter Plot related functions
# ==============================
def gen_scatter(dataset,data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_linear_fit=False, ax_lim=None, hexbin=False, title=None):
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
    scat           = data_df.hvplot.scatter(x=eep1, y=eep2, aspect='square',s=1, xlim=lims, ylim=lims, alpha=.7, title=title) #.opts(active_tools=['save'], tools=['save'])
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
     
    return plot.opts(active_tools=['reset'])


def scan_summary_tab(sbj,ses,pp, nordic,fc_metric, 
                        data_scat,data_qc):
    """
    Create panel Frame with scatter plots for all FC combinations and table with QC metrics

    Inputs:
    -------
    data_scat (xarray.DataArray): xarray object with quality metrics
    sbj (str): subject ID
    ses (str): session ID
    pp (str): pre-processing pipeline
    nordic (str): whether Nordic was used or not
    fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
    other_stats (pd.DataFrame or None): additional statistics to be displayed in the table

    Returns:
    --------
    pn.Row: Panel layout containing the scatter plots and statistics table
    """
    
    # TSNR Card
    # =========
    stats_df      = data_scat.loc[sbj,ses,pp,nordic,fc_metric,:,:].to_dataframe(name='QC').reset_index().drop(['sbj','ses','pp','fc_metric'],axis=1).pivot(index='ee_vs_ee', columns='qc_metric', values='QC')
    stats_mean_df = pd.DataFrame(stats_df.mean(),columns=['Avg']).T
    stats_mean_df.loc['Weighted Avg','pBOLD'] = data_qc[fc_metric,'pBOLD'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
    stats_mean_df.loc['Weighted Avg','pSo']   = data_qc[fc_metric,'pSo'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
    
    TSNR_df                         = pd.DataFrame(columns=['Full Brain','Visual Cortex'],index=[pp])
    TSNR_df.loc[pp,'Full Brain']    = data_qc[fc_metric,'TSNR (Full Brain)'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
    TSNR_df.loc[pp,'Visual Cortex'] = data_qc[fc_metric,'TSNR (Visual Cortex)'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
    TSNR_df_ready_to_plot           = data_qc[fc_metric,'TSNR (Full Brain)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
    TSNR_df_ready_to_plot           = TSNR_df_ready_to_plot.replace(LABEL_MAPPING)
    TSNR_plot                       = TSNR_df_ready_to_plot.hvplot.bar(x='Pre-processing',y='TSNR (Full Brain)', width=300).opts(xrotation=90, toolbar=None)
    TSNR_card                       = pn.Card(pn.Column(TSNR_df, TSNR_plot), title='TSNR', width=350)

    # pBOLD Card
    # ===========
    pBOLD_df                    = pd.DataFrame(columns=['pBOLD'],index=[pp])
    pBOLD_df.loc[pp,'pBOLD']    = data_qc[fc_metric,'pBOLD'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
    pBOLD_df_ready_to_plot      = data_qc[fc_metric,'pBOLD'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
    pBOLD_df_ready_to_plot      = pBOLD_df_ready_to_plot.replace(LABEL_MAPPING)
    pBOLD_plot                  = pBOLD_df_ready_to_plot.hvplot.bar(x='Pre-processing',y='pBOLD', width=300).opts(xrotation=90, toolbar=None)
    #pBOLD_card                  = pn.Card(pn.Column(pBOLD_df, pBOLD_plot), title='pBOLD', width=350)
    pBOLD_card                  = pn.Card(pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2)),pBOLD_plot), title='pBOLD', width=350)

    # Tedana Card
    # ===========
    ics_all_df    = data_qc['C','#ICs (All)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
    ics_lbold_df  = data_qc['C','#ICs (Likely BOLD)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
    ics_ulbold_df = data_qc['C','#ICs (Unlikely BOLD)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
    
    ics_all_df    = ics_all_df.replace(LABEL_MAPPING)
    ics_lbold_df  = ics_lbold_df.replace(LABEL_MAPPING)
    ics_ulbold_df = ics_ulbold_df.replace(LABEL_MAPPING)
    tedana_df = pd.concat([ics_all_df.set_index('Pre-processing'), ics_lbold_df.set_index('Pre-processing'), ics_ulbold_df.set_index('Pre-processing')],axis=1).dropna()
    tedana_df.columns = ['All','BOLD','unl BOLD']
    tedana_plot = tedana_df.hvplot.bar(x='Pre-processing', width=300).opts(xrotation=90, toolbar=None)
    tedana_card                       = pn.Card(pn.Column(tedana_df, tedana_plot), title='Tedana Components', width=350)

    tables = pn.Tabs(('pBOLD',pBOLD_card),('TSNR',TSNR_card),('Tedana',tedana_card))

    # Physio Information
    # ==================
    resp_plot, card_plot = None,None
    if data_qc['C',('Physio (resp)')] is not None:
        if (sbj,ses) in data_qc['C',('Physio (resp)')].index:
            resp_plot = data_qc['C',('Physio (resp)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Card. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
            resp_plot = resp_plot * hv.Points((data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
            resp_plot = resp_plot.opts(shared_axes=False)
        if (sbj,ses) in data_qc['C',('Physio (cardiac)')].index:
            card_plot = data_qc['C',('Physio (cardiac)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Resp. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
            card_plot = card_plot * hv.Points((data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
            card_plot = card_plot.opts(shared_axes=False)
    else:
        resp_plot,card_plot = None,None
    physio_card = pn.Card(pn.Column(resp_plot,card_plot),title='Physio',width=350)

    tables = pn.Tabs(('pBOLD',pBOLD_card),('TSNR',TSNR_card),('Tedana',tedana_card),('Physio',physio_card))

    return tables

def fc_across_echoes_scatter_page(dataset,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, 
                                  data_fc,data_scat,data_qc,
                                  show_line=False, ax_lim=None, hexbin=False):
    """
    Create panel Frame with scatter plots for all FC combinations and table with QC metrics

    Inputs:
    -------
    dataset (str): name of the dataset being used (e.g., evaluation, discovery)
    data_fc (dict): dictionary with FC matrices
    data_scat (xarray.DataArray): xarray object with quality metrics
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
    # =====================
    scatter_layout = pn.layout.GridBox(ncols=5)
    for i in pairs_of_echo_pairs:
        eep1,eep2=i.split('_vs_')
        this_scatter_pBOLD = data_scat.sel(sbj=sbj,ses=ses,fc_metric=fc_metric,ee_vs_ee=i,nordic=nordic,pp=pp,qc_metric='pBOLD').values
        this_scatter_pSo   = data_scat.sel(sbj=sbj,ses=ses,fc_metric=fc_metric,ee_vs_ee=i,nordic=nordic,pp=pp,qc_metric='pSo').values
        #this_scatter_TSNR  = data_qc[fc_metric,'TSNR (Full Brain)'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
        title = 'pBOLD=%.2f | pSo=%.2f' % (this_scatter_pBOLD, this_scatter_pSo)
        plot = gen_scatter(dataset,data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_line, ax_lim, hexbin=hexbin, title=title)
        scatter_layout.append(plot)

    tables = scan_summary_tab(sbj,ses,pp, nordic,fc_metric, 
                        data_scat,data_qc)
    # Create Page
    # ===========
    frame = pn.Row(scatter_layout, tables)
    return frame
    
def get_gs_detailed_view(gs_df_spc, ic_df_ts,card_df,resp_df,rvt_df,card_reg_df,resp_reg_df):
    layout = pn.Column()
    # Generate GS plot
    opts_kwords = {'xlabel':"MM:SS", 'show_grid':True, 'width':1500 ,'xticks':np.arange(30000,609000,30000), 'xformatter':DatetimeTickFormatter(minutes="%M:%S", seconds="%M:%S"), 'show_legend':True,'active_tools':[],'fontscale':1.2}
    gs_opts_kwords = opts_kwords.copy(); opts_kwords['title']='Global Signal'
    oc_opts_kwords = gs_opts_kwords.copy(); oc_opts_kwords['color']="#0000ff"; oc_opts_kwords['line_dash']="dashed"; oc_opts_kwords['line_width']=0.5
    e1_opts_kwords = gs_opts_kwords.copy(); e1_opts_kwords['color']="#4d4d4d"
    e2_opts_kwords = gs_opts_kwords.copy(); e2_opts_kwords['color']="#7f7f7f"
    e3_opts_kwords = gs_opts_kwords.copy(); e3_opts_kwords['color']="#b3b3b3"
    gs_plot = hv.Curve((gs_df_spc.index,gs_df_spc['OC']),vdims=[hv.Dimension('Sig. % Change')], label='OC').opts(**oc_opts_kwords) * \
              hv.Curve((gs_df_spc.index,gs_df_spc['TE1']),vdims=[hv.Dimension('Sig. % Change')], label='TE1').opts(**e1_opts_kwords) * \
              hv.Curve((gs_df_spc.index,gs_df_spc['TE2']),vdims=[hv.Dimension('Sig. % Change')], label='TE2').opts(**e2_opts_kwords) * \
              hv.Curve((gs_df_spc.index,gs_df_spc['TE3']),vdims=[hv.Dimension('Sig. % Change')], label='TE3').opts(**e3_opts_kwords)
    layout.append(gs_plot)
    # IC Timeseries
    ica_opts_kwords = opts_kwords.copy(); opts_kwords['title']='ICs corr w/ GS'
    ica_plot = None
    for i,ic in enumerate(ic_df_ts.columns):
        if ica_plot is None:
            ica_plot = hv.Curve((ic_df_ts.index,ic_df_ts[ic]),vdims=[hv.Dimension('Sig. % Change')], label=f'{ic} | Rank = {i}').opts(**ica_opts_kwords)
        else:
            ica_plot = ica_plot * hv.Curve((ic_df_ts.index,ic_df_ts[ic]),vdims=[hv.Dimension('Sig. % Change')], label=f'{ic} | Rank = {i}').opts(**ica_opts_kwords)
    layout.append(ica_plot)
    # Physiological Signals
    if card_df is not None:
        preg_opts_kwords = opts_kwords.copy(); opts_kwords['title']='Physio Traces'
        card_opts_kwords = preg_opts_kwords.copy(); card_opts_kwords['yaxis']='right'
        resp_opts_kwords = preg_opts_kwords.copy(); resp_opts_kwords['yaxis']='left'
        preg_plot = (hv.Curve((card_df.index,card_df['PGG']),vdims=[hv.Dimension('PGG')], label='PGG').opts(**card_opts_kwords) * \
                    hv.Curve((resp_df.index,resp_df['Respiration']),vdims=[hv.Dimension('Respiration')], label='Respiration').opts(**resp_opts_kwords)).opts(multi_y=True)
        layout.append(preg_plot)
    # Physiological Regeressors
    for physio_reg_df,physio_reg_title,physio_reg_units in zip([rvt_df,resp_reg_df,card_reg_df],
                                             ['Physio Regressors (RVT)','Physio Regressors (Respiration)','Physio Regressors (cardiac)'],
                                             ['RVT','Resp. Reg.','Card. Reg.']):
        if physio_reg_df is not None:
            preg_opts_kwords = opts_kwords.copy(); preg_opts_kwords['title']=physio_reg_title
            preg_plot = None
            for col in physio_reg_df.columns:
                if preg_plot is None:
                    preg_plot = hv.Curve((physio_reg_df.index,physio_reg_df[col]),vdims=[hv.Dimension(physio_reg_units)], label=col).opts(**preg_opts_kwords)
                else:
                    preg_plot = preg_plot * hv.Curve((physio_reg_df.index,physio_reg_df[col]),vdims=[hv.Dimension(physio_reg_units)], label=col).opts(**preg_opts_kwords)
            layout.append(preg_plot)
    # Respiration once again on its own
    resp_opts_kwords = opts_kwords.copy(); resp_opts_kwords['title']='Respiration'; resp_opts_kwords['line_color']='black'; resp_opts_kwords['show_legend']=False
    resp_only_plot = hv.Curve((resp_df.index,resp_df['Respiration']),vdims=[hv.Dimension('Respiration')]).opts(**resp_opts_kwords)
    layout.append(resp_only_plot)
    return layout

def get_ts_report_page(sbj,ses,pp,nordic,fc_metric,num_ics_to_show,
                       gs_df_dict,ica_dict,physio_dict,physio_reg_dict,data_scat,data_qc):
    gs_df_spc   = gs_df_dict[sbj,ses,'gs_ts']
    ic_df_ts    = ica_dict[sbj,ses,'ic_ts']
    ics_to_show = ic_df_ts.corrwith(gs_df_spc['OC']).abs().sort_values(ascending=False).index[0:num_ics_to_show]
    ic_df_ts    = ic_df_ts[ics_to_show]
    card_df     = physio_dict[sbj,ses,'card']
    resp_df     = physio_dict[sbj,ses,'resp']
    rvt_df      = physio_reg_dict[(sbj,ses,'RVT_regs')]
    card_reg_df = physio_reg_dict[(sbj,ses,'card_regs')]
    resp_reg_df = physio_reg_dict[(sbj,ses,'resp_regs')]
    gs_info_df  = gs_df_dict[sbj,ses,'gs_metrics']
    ica_info_df = ica_dict[sbj,ses,'ic_metrics']
    ica_info_df = ica_info_df.loc[list(ics_to_show)].round(2)
    basic_side_tab = scan_summary_tab(sbj,ses,pp,nordic,fc_metric,data_scat,data_qc)
    other_card     = pn.Card(pn.Column(pn.pane.DataFrame(gs_info_df),pn.pane.DataFrame(ica_info_df)),title='Other',width=350)
    basic_side_tab.append(('Other',other_card))
    return pn.Row(pn.Column(get_gs_detailed_view(gs_df_spc,ic_df_ts,card_df,resp_df,rvt_df,card_reg_df,resp_reg_df)),basic_side_tab)
    
# def fc_across_echoes_scatter_page(dataset,sbj,ses,pp, nordic,fc_metric, pairs_of_echo_pairs, 
#                                   data_fc,data_scat,data_qc,
#                                   show_line=False, ax_lim=None, hexbin=False):
#     """
#     Create panel Frame with scatter plots for all FC combinations and table with QC metrics

#     Inputs:
#     -------
#     dataset (str): name of the dataset being used (e.g., evaluation, discovery)
#     data_fc (dict): dictionary with FC matrices
#     data_scat (xarray.DataArray): xarray object with quality metrics
#     sbj (str): subject ID
#     ses (str): session ID
#     pp (str): pre-processing pipeline
#     nordic (str): whether Nordic was used or not
#     fc_metric (str): FC metric to be used ('R' for correlation, 'C' for covariance)
#     pairs_of_echo_pairs (list): list of echo pair combinations in the format ['e02|e02_vs_e03|e03', ...]
#     show_line (bool): whether to show linear fit line or not
#     ax_lim (float): axis limits for the scatter plot
#     other_stats (pd.DataFrame or None): additional statistics to be displayed in the table
#     hexbin (bool): whether to use hexbin plot instead of scatter plot

#     Returns:
#     --------
#     pn.Row: Panel layout containing the scatter plots and statistics table
#     """

#     # Grid of scatter plots
#     # =====================
#     scatter_layout = pn.layout.GridBox(ncols=5)
#     for i in pairs_of_echo_pairs:
#         eep1,eep2=i.split('_vs_')
#         this_scatter_pBOLD = data_scat.sel(sbj=sbj,ses=ses,fc_metric=fc_metric,ee_vs_ee=i,nordic=nordic,pp=pp,qc_metric='pBOLD').values
#         this_scatter_pSo   = data_scat.sel(sbj=sbj,ses=ses,fc_metric=fc_metric,ee_vs_ee=i,nordic=nordic,pp=pp,qc_metric='pSo').values
#         #this_scatter_TSNR  = data_qc[fc_metric,'TSNR (Full Brain)'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
#         title = 'pBOLD=%.2f | pSo=%.2f' % (this_scatter_pBOLD, this_scatter_pSo)
#         plot = gen_scatter(dataset,data_fc,sbj,ses,pp,nordic,eep1,eep2,fc_metric, show_line, ax_lim, hexbin=hexbin, title=title)
#         scatter_layout.append(plot)

#     # TSNR Card
#     # =========
#     stats_df      = data_scat.loc[sbj,ses,pp,nordic,fc_metric,:,:].to_dataframe(name='QC').reset_index().drop(['sbj','ses','pp','fc_metric'],axis=1).pivot(index='ee_vs_ee', columns='qc_metric', values='QC')
#     stats_mean_df = pd.DataFrame(stats_df.mean(),columns=['Avg']).T
#     stats_mean_df.loc['Weighted Avg','pBOLD'] = data_qc[fc_metric,'pBOLD'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
#     stats_mean_df.loc['Weighted Avg','pSo']   = data_qc[fc_metric,'pSo'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
    
#     TSNR_df                         = pd.DataFrame(columns=['Full Brain','Visual Cortex'],index=[pp])
#     TSNR_df.loc[pp,'Full Brain']    = data_qc[fc_metric,'TSNR (Full Brain)'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
#     TSNR_df.loc[pp,'Visual Cortex'] = data_qc[fc_metric,'TSNR (Visual Cortex)'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
#     TSNR_df_ready_to_plot           = data_qc[fc_metric,'TSNR (Full Brain)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
#     TSNR_df_ready_to_plot           = TSNR_df_ready_to_plot.replace(LABEL_MAPPING)
#     TSNR_plot                       = TSNR_df_ready_to_plot.hvplot.bar(x='Pre-processing',y='TSNR (Full Brain)', width=300).opts(xrotation=90, toolbar=None)
#     TSNR_card                       = pn.Card(pn.Column(TSNR_df, TSNR_plot), title='TSNR', width=350)

#     # pBOLD Card
#     # ===========
#     pBOLD_df                    = pd.DataFrame(columns=['pBOLD'],index=[pp])
#     pBOLD_df.loc[pp,'pBOLD']    = data_qc[fc_metric,'pBOLD'].set_index(['Subject','Session','Pre-processing','NORDIC']).loc[sbj,ses,pp,nordic].values[0]
#     pBOLD_df_ready_to_plot      = data_qc[fc_metric,'pBOLD'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
#     pBOLD_df_ready_to_plot      = pBOLD_df_ready_to_plot.replace(LABEL_MAPPING)
#     pBOLD_plot                  = pBOLD_df_ready_to_plot.hvplot.bar(x='Pre-processing',y='pBOLD', width=300).opts(xrotation=90, toolbar=None)
#     #pBOLD_card                  = pn.Card(pn.Column(pBOLD_df, pBOLD_plot), title='pBOLD', width=350)
#     pBOLD_card                  = pn.Card(pn.Column(pn.pane.DataFrame(stats_df.round(2)), pn.layout.Divider(), pn.pane.DataFrame(stats_mean_df.round(2)),pBOLD_plot), title='pBOLD', width=350)

#     # Tedana Card
#     # ===========
#     ics_all_df    = data_qc['C','#ICs (All)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
#     ics_lbold_df  = data_qc['C','#ICs (Likely BOLD)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
#     ics_ulbold_df = data_qc['C','#ICs (Unlikely BOLD)'].set_index(['Subject','Session','NORDIC','Pre-processing']).loc[sbj,ses,nordic,:].reset_index()
    
#     ics_all_df    = ics_all_df.replace(LABEL_MAPPING)
#     ics_lbold_df  = ics_lbold_df.replace(LABEL_MAPPING)
#     ics_ulbold_df = ics_ulbold_df.replace(LABEL_MAPPING)
#     tedana_df = pd.concat([ics_all_df.set_index('Pre-processing'), ics_lbold_df.set_index('Pre-processing'), ics_ulbold_df.set_index('Pre-processing')],axis=1).dropna()
#     tedana_df.columns = ['All','BOLD','unl BOLD']
#     tedana_plot = tedana_df.hvplot.bar(x='Pre-processing', width=300).opts(xrotation=90, toolbar=None)
#     tedana_card                       = pn.Card(pn.Column(tedana_df, tedana_plot), title='Tedana Components', width=350)

#     tables = pn.Tabs(('pBOLD',pBOLD_card),('TSNR',TSNR_card),('Tedana',tedana_card))

#     # Physio Information
#     # ==================
#     resp_plot, card_plot = None,None
#     if (sbj,ses) in data_qc['C',('Physio (resp)')].index:
#         resp_plot = data_qc['C',('Physio (resp)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Card. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
#         resp_plot = resp_plot * hv.Points((data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (resp)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
#         resp_plot = resp_plot.opts(shared_axes=False)
#     if (sbj,ses) in data_qc['C',('Physio (cardiac)')].index:
#         card_plot = data_qc['C',('Physio (cardiac)')].hvplot.scatter(x='Mean',y='St.Dev.', c='color', title='Resp. Inter-peak Interval', aspect='square', hover_cols=['Subject','Run'], alpha=0.5, width=300, height=300) 
#         card_plot = card_plot * hv.Points((data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['Mean'],data_qc['C',('Physio (cardiac)')].loc[(sbj,ses)]['St.Dev.'])).opts(size=10, marker="o",line_color="black", line_width=2,fill_alpha=0.0)
#         card_plot = card_plot.opts(shared_axes=False)

#     physio_card = pn.Card(pn.Column(resp_plot,card_plot),title='Physio',width=350)

#     tables = pn.Tabs(('pBOLD',pBOLD_card),('TSNR',TSNR_card),('Tedana',tedana_card),('Physio',physio_card))

#     #tables = pn.Column(pBOLD_card, pn.Tabs((TSNR_card, tedana_card))
    
#     # Create Page
#     # ===========
#     frame = pn.Row(scatter_layout, tables)
#     return frame
    
    

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
    pp_avail = list(qa_xr.coords['pp'].values)
    for pp, pp_label in zip(pp_avail,[LABEL_MAPPING[p] for p in pp_avail]):
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
def get_static_report(qa,fc_metric,qc_metric,x='Pre-processing',hue='NORDIC',show_stats=False, stat_test='t-test_paired',stat_annot_type='star', legend_location='best', remove_outliers_from_swarm=True, remove_outliers_from_statistics=False, dot_size=1, palette='Set2', show_points=False,session='all', label_font_size=12, pair_weights=None):
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
        if pair_weights is None:
            df         = qa.mean(dim='ee_vs_ee').sel(fc_metric=fc_metric, qc_metric=qc_metric).to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric'],axis=1).reset_index()
        else:
            df         = qa.weighted(pair_weights).mean(dim='ee_vs_ee').sel(fc_metric='C', qc_metric='pBOLD').to_dataframe(name='C').drop(['fc_metric','qc_metric'],axis=1).reset_index()
        df.columns = ['Subject','Session','Pre-processing','NORDIC',qc_metric]
    elif isinstance(qa, pd.DataFrame):
        df = qa.copy()
    else:
        print("++ ERROR [get_barplot]: Expected a xarray or pandas dataframe but got something else. Function exiting")
        return None
    df         = df.replace(LABEL_MAPPING)
    if session !='all':
        df = df[df['Session']==session]
    # Extract available values for X and HUE
    x_options   = list(df[x].unique())
    hue_options = list(df[hue].unique())
    num_x_categories = len(x_options)
    num_hue_categories = len(hue_options)


    # Calculate Q1, Q3, and IQR
    Q1 = df[qc_metric].quantile(0.25)
    Q3 = df[qc_metric].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_swarm = df.copy()
    if remove_outliers_from_swarm:
        df_swarm = df_swarm[(df_swarm[qc_metric] >= lower_bound) & (df_swarm[qc_metric] <= upper_bound)]
        #quantile_value = df[qc_metric].quantile(.97)
        #df_swarm[qc_metric]=df_swarm[qc_metric].where(df_swarm[qc_metric] <= quantile_value, np.nan)
        #df_swarm.dropna(inplace=True)
    if remove_outliers_from_statistics:
        df = df[(df[qc_metric] >= lower_bound) & (df[qc_metric] <= upper_bound)]
        
    pairs  = [((x,h[1]),(x,h[0])) for x in x_options for h in combinations(hue_options,2)]
    colors = sns.color_palette(palette,num_hue_categories) 

    sns.set_context("paper", rc={"xtick.labelsize": label_font_size, "ytick.labelsize": label_font_size, "axes.labelsize": label_font_size, 'legend.fontsize':label_font_size})
    fig, axs = plt.subplots(1,1,figsize=(6,6));
    sns.despine(top=True, right=True)
    sns.barplot(data=df,hue=hue, y=qc_metric, x=x, alpha=0.5, ax =axs, errorbar=('ci',95), palette=colors);
    if show_points:
        sns.swarmplot(data=df_swarm,hue=hue, y=qc_metric, x=x, ax =axs, s=dot_size, edgecolor="black", linewidth=0.5, dodge=True, legend=False, palette=colors, alpha=0.5);
    
    if show_stats:
        annotation = Annotator(axs, pairs, data=df, x=x, y=qc_metric, hue=hue);
        annotation.configure(test=stat_test, text_format=stat_annot_type, loc='inside', verbose=0);
        annotation.apply_test(alternative='two-sided');
        annotation.annotate();
    if num_hue_categories > 3:
        ncol = 2
    else:
        ncol = num_hue_categories
    sns.move_legend(axs, "lower center", bbox_to_anchor=(.5, 1), ncol=ncol, title=None, frameon=False)
    plt.tight_layout()
    plt.close()
    return fig
    

# Dynamic Group Level Report Functions
# ====================================
def dynamic_summary_plot_gated(qa_xr, fc_metric, qc_metric, nordic):
    df= qa_xr.sel(fc_metric=fc_metric, nordic=nordic, qc_metric=qc_metric).mean(dim='ee_vs_ee').to_dataframe(name=qc_metric).drop(['fc_metric','qc_metric','nordic'],axis=1).reset_index()
    df.columns=['Subject','Session','Pre-processing',qc_metric]
    df = df.replace(LABEL_MAPPING)
    df['Scenario'] = df['Session']+'\n'+df['Pre-processing']

    plot = df.hvplot.box(   by='Scenario',y=qc_metric, legend=False) * \
           df.hvplot.scatter(x='Scenario',y=qc_metric, by='Subject', legend=False, hover_cols=['Subject','Session']) * \
           df.hvplot.line(by=['Subject','Session'],x='Scenario',y=qc_metric, legend=False, c='k', line_dash='dashed', line_width=0.5)
    return plot.opts(legend_position='top', height=400, width=600, legend_cols=4)