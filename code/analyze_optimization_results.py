#%%

import numpy as np
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from numpy import newaxis as na

from matplotlib.lines import Line2D


import xarray as xr

#%%
# update font to latex
try:
    import matplotlib.font_manager as font_manager
    font_dir = [r"input_folder\font\Serif"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({
        # Regular text
        "font.family": "CMU Serif",
    
        # Math text
        "mathtext.fontset": "custom",
        "mathtext.rm": "CMU Serif",
        "mathtext.it": "CMU Serif:italic",
        "mathtext.bf": "CMU Serif:bold",
        "mathtext.sf": "CMU Serif",
        "mathtext.tt": "CMU Serif",
    
        # Your existing options
        "svg.fonttype": "path",
        "axes.unicode_minus": False,
    })
except ValueError:
    print("Latex font not available.")

#%%
# functions

def convert_ds_fromMOtoSO(ds_MO,obj,maximize=True):
    best_list = []
    for n_sample, ds_g in ds_MO.groupby('n_sample'):
        if maximize:
            best_id = ds_g[obj].argmax('id')
        else:
            best_id = ds_g[obj].argmin('id')
        best = ds_g.isel(id=best_id).expand_dims(n_sample=[n_sample])
        best_list.append(best)
    ds_SO = xr.concat(best_list, dim='n_sample')
    return ds_SO


def filter_ds(ds,
              case_study,
              obj,
              maximize=True,
              convert_to_SO=False):
    
    if obj=='LCOE':
        obj_ds = 'IOE'
    elif obj=='LCOE_GY':
        obj_ds = 'IOE_GY'
    else:
        obj_ds = obj

    if obj.endswith('_GY'):
        gy_ds = '_GY'
        obj_ds = obj_ds[:-3]
        if obj_ds=='AEPnet':
            obj_ds = 'AEPn'
    else:
        gy_ds = ''
        
    if obj.endswith('_GY'):
        obj_val = obj[:-3]+'_OY'
    else:
        obj_val = obj
        
    ds_fil = ds.where((ds['obj']==f'MO{gy_ds}_LCOEopt_{obj_ds}opt')&(ds['case_study']==case_study),drop=True)

    if convert_to_SO:
        best_list = []
        for n_sample, ds_g in ds_fil.groupby('n_sample'):
            if maximize:
                best_id = ds_g[obj_val].argmax('id')
            else:
                best_id = ds_g[obj_val].argmin('id')
            best = ds_g.isel(id=best_id).expand_dims(n_sample=[n_sample])
            best_list.append(best)
        ds_SO = xr.concat(best_list, dim='n_sample')
        return ds_SO
    else:
        return ds_fil


#%%
# load data

with open('optimization_results\\20260727_ds_CabYaw_all.pkl','rb') as f: 
    data = pickle.load(f)
ds_all = data['ds']


# separate objectives

# HKNscaled
ds_HKNs_AEP = filter_ds(ds_all,case_study='HKNscaled',obj='AEP',convert_to_SO=True,maximize=True)
ds_HKNs_AEPnet = filter_ds(ds_all,case_study='HKNscaled',obj='AEPnet',convert_to_SO=True,maximize=True)
ds_HKNs_LCOE = filter_ds(ds_all,case_study='HKNscaled',obj='LCOE',convert_to_SO=True,maximize=False)
ds_HKNs_IOE = filter_ds(ds_all,case_study='HKNscaled',obj='IOE',convert_to_SO=True,maximize=False)
ds_HKNs_MO = filter_ds(ds_all,case_study='HKNscaled',obj='IOE',convert_to_SO=False,maximize=False)
ds_HKNs_AEP_GY = filter_ds(ds_all,case_study='HKNscaled',obj='AEP_GY',convert_to_SO=True,maximize=True)
ds_HKNs_AEPnet_GY = filter_ds(ds_all,case_study='HKNscaled',obj='AEPnet_GY',convert_to_SO=True,maximize=True)
ds_HKNs_LCOE_GY = filter_ds(ds_all,case_study='HKNscaled',obj='LCOE_GY',convert_to_SO=True,maximize=False)
ds_HKNs_IOE_GY = filter_ds(ds_all,case_study='HKNscaled',obj='IOE_GY',convert_to_SO=True,maximize=False)
ds_HKNs_MO_GY = filter_ds(ds_all,case_study='HKNscaled',obj='IOE_GY',convert_to_SO=False,maximize=False)

# HKNscaled - LPD
ds_HKNsLPD_LCOE = filter_ds(ds_all,case_study='HKNsLPD',obj='LCOE',convert_to_SO=True,maximize=False)
ds_HKNsLPD_IOE = filter_ds(ds_all,case_study='HKNsLPD',obj='IOE',convert_to_SO=True,maximize=False)
ds_HKNsLPD_MO = filter_ds(ds_all,case_study='HKNsLPD',obj='IOE',convert_to_SO=False,maximize=False)
ds_HKNsLPD_LCOE_GY = filter_ds(ds_all,case_study='HKNsLPD',obj='LCOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsLPD_IOE_GY = filter_ds(ds_all,case_study='HKNsLPD',obj='IOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsLPD_MO_GY = filter_ds(ds_all,case_study='HKNsLPD',obj='IOE_GY',convert_to_SO=False,maximize=False)

# HKNscaled - HPD
ds_HKNsHPD_LCOE = filter_ds(ds_all,case_study='HKNsHPD',obj='LCOE',convert_to_SO=True,maximize=False)
ds_HKNsHPD_IOE = filter_ds(ds_all,case_study='HKNsHPD',obj='IOE',convert_to_SO=True,maximize=False)
ds_HKNsHPD_MO = filter_ds(ds_all,case_study='HKNsHPD',obj='IOE',convert_to_SO=False,maximize=False)
ds_HKNsHPD_LCOE_GY = filter_ds(ds_all,case_study='HKNsHPD',obj='LCOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsHPD_IOE_GY = filter_ds(ds_all,case_study='HKNsHPD',obj='IOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsHPD_MO_GY = filter_ds(ds_all,case_study='HKNsHPD',obj='IOE_GY',convert_to_SO=False,maximize=False)

# HKNscaled - HBD
ds_HKNsHBD_LCOE = filter_ds(ds_all,case_study='HKNsHBD',obj='LCOE',convert_to_SO=True,maximize=False)
ds_HKNsHBD_IOE = filter_ds(ds_all,case_study='HKNsHBD',obj='IOE',convert_to_SO=True,maximize=False)
ds_HKNsHBD_MO = filter_ds(ds_all,case_study='HKNsHBD',obj='IOE',convert_to_SO=False,maximize=False)
ds_HKNsHBD_LCOE_GY = filter_ds(ds_all,case_study='HKNsHBD',obj='LCOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsHBD_IOE_GY = filter_ds(ds_all,case_study='HKNsHBD',obj='IOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsHBD_MO_GY = filter_ds(ds_all,case_study='HKNsHBD',obj='IOE_GY',convert_to_SO=False,maximize=False)

# HKNscaled - HSBD
ds_HKNsHSBD_LCOE = filter_ds(ds_all,case_study='HKNsHSBD',obj='LCOE',convert_to_SO=True,maximize=False)
ds_HKNsHSBD_IOE = filter_ds(ds_all,case_study='HKNsHSBD',obj='IOE',convert_to_SO=True,maximize=False)
ds_HKNsHSBD_MO = filter_ds(ds_all,case_study='HKNsHSBD',obj='IOE',convert_to_SO=False,maximize=False)
ds_HKNsHSBD_LCOE_GY = filter_ds(ds_all,case_study='HKNsHSBD',obj='LCOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsHSBD_IOE_GY = filter_ds(ds_all,case_study='HKNsHSBD',obj='IOE_GY',convert_to_SO=True,maximize=False)
ds_HKNsHSBD_MO_GY = filter_ds(ds_all,case_study='HKNsHSBD',obj='IOE_GY',convert_to_SO=False,maximize=False)

#%%

from functools import partial

def y_to_pct(y, ref_y):
    return 100*((y-ref_y)/ref_y)

def pct_to_y(p, ref_y):
    return ((p+ref_y)/100)*ref_y


def boxplot_codesign_v3(axs,
                        ds_LCOE,
                        ds_IOE,
                        ds_LCOE_GY,
                        ds_IOE_GY,
                        colors
                        ):

    #colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887','#4dc064']
    x_plot = np.arange(3)
    width = 0.7
    flierprops = dict(marker='o',markersize=2,markerfacecolor='black',markeredgecolor='black',linestyle='none')

    # LCOE - define reference value and create a secondary axis
    ind_best_LCOE = np.array(ds_LCOE['LCOE']).argmin()
    ref_y = ds_LCOE['LCOE'][ind_best_LCOE].values
    y_to_pct_LCOE = partial(y_to_pct, ref_y=ref_y)
    pct_to_y_LCOE = partial(pct_to_y, ref_y=ref_y)
    secax_y = axs[0].secondary_yaxis('right', functions=(y_to_pct_LCOE, pct_to_y_LCOE))
    axs[0].axhline(ref_y, color='k', linestyle='solid', linewidth=1,zorder=-2,alpha=0.5)
    axs[0].axhline(np.min(ds_LCOE['LCOE_OY']), color='k', linestyle='dashed', linewidth=1,zorder=-2,alpha=0.5)
    axs[0].axhline(np.min(ds_LCOE_GY['LCOE_OY']), color='k', linestyle='dotted', linewidth=1,zorder=-2,alpha=0.5)

    # layout optimization
    bp1 = axs[0].boxplot([ds_LCOE['LCOE']],positions=[x_plot[0]],widths=width,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp1['boxes'][0].set_facecolor(colors[0])

    # control optimization
    bp1 = axs[0].boxplot([ds_LCOE['LCOE_OY']],positions=[x_plot[1]],widths=width,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp1['boxes'][0].set_facecolor(colors[1])

    # co-design
    bp1 = axs[0].boxplot([ds_LCOE_GY['LCOE_OY']],positions=[x_plot[2]],widths=width,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp1['boxes'][0].set_facecolor(colors[2])

    # IOE - define reference value and create a secondary axis
    ind_best_IOE = np.array(ds_IOE['IOE']).argmin()
    ref_y2 = ds_IOE['IOE'][ind_best_IOE].values
    y_to_pct_IOE = partial(y_to_pct, ref_y=ref_y2)
    pct_to_y_IOE = partial(pct_to_y, ref_y=ref_y2)
    secax_y2 = axs[1].secondary_yaxis('right', functions=(y_to_pct_IOE, pct_to_y_IOE))
    axs[1].axhline(ref_y2, color='k', linestyle='solid', linewidth=1,zorder=-2,alpha=0.5)
    axs[1].axhline(np.min(ds_IOE['IOE_OY']), color='k', linestyle='dashed', linewidth=1,zorder=-2,alpha=0.5)
    axs[1].axhline(np.min(ds_IOE_GY['IOE_OY']), color='k', linestyle='dotted', linewidth=1,zorder=-2,alpha=0.5)

    # layout optimization
    bp2 = axs[1].boxplot([ds_IOE['IOE']],positions=[x_plot[0]],widths=width,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp2['boxes'][0].set_facecolor(colors[0])

    # control optimization
    bp2 = axs[1].boxplot([ds_IOE['IOE_OY']],positions=[x_plot[1]],widths=width,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp2['boxes'][0].set_facecolor(colors[1])

    # co-design
    bp2 = axs[1].boxplot([ds_IOE_GY['IOE_OY']],positions=[x_plot[2]],widths=width,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp2['boxes'][0].set_facecolor(colors[2])

    plt.tight_layout()
    axs[0].set_xticks([])
    axs[1].set_xticks([])

    print('-------------')
    print(f"LCOE: {np.median(y_to_pct_LCOE(ds_LCOE['LCOE_OY'].values))}")
    print(f"IOE: {np.median(y_to_pct_IOE(ds_IOE['IOE_OY'].values))}")
    print(f"LCOE GY: {np.median(y_to_pct_LCOE(ds_LCOE_GY['LCOE_OY'].values))}")
    print(f"IOE GY: {np.median(y_to_pct_IOE(ds_IOE_GY['IOE_OY'].values))}")

    return axs

savefig = False
formatfig = 'pdf'
pathfig = './'


colors = ['#0072B2', '#E69F00', '#009E73']

fig,axs = plt.subplots(figsize=(10,6),nrows=2,ncols=5,sharex=True)
axs0 = boxplot_codesign_v3(axs[:,0],ds_HKNs_LCOE,ds_HKNs_IOE,ds_HKNs_LCOE_GY,ds_HKNs_IOE_GY,colors)
#axs1 = boxplot_codesign_v3(axs[:,1],ds_HKNsHPD_LCOE,ds_HKNsHPD_IOE,ds_HKNsHPD_LCOE_GY,ds_HKNsHPD_IOE_GY,colors)
axs1 = boxplot_codesign_v3(axs[:,1],ds_HKNsLPD_LCOE,ds_HKNsLPD_IOE,ds_HKNsLPD_LCOE_GY,ds_HKNsLPD_IOE_GY,colors)
axs2 = boxplot_codesign_v3(axs[:,2],ds_HKNsHPD_LCOE,ds_HKNsHPD_IOE,ds_HKNsHPD_LCOE_GY,ds_HKNsHPD_IOE_GY,colors)
axs3 = boxplot_codesign_v3(axs[:,3],ds_HKNsHBD_LCOE,ds_HKNsHBD_IOE,ds_HKNsHBD_LCOE_GY,ds_HKNsHBD_IOE_GY,colors)
axs4 = boxplot_codesign_v3(axs[:,4],ds_HKNsHSBD_LCOE,ds_HKNsHSBD_IOE,ds_HKNsHSBD_LCOE_GY,ds_HKNsHSBD_IOE_GY,colors)
axs0[0].set_title('BL')
axs1[0].set_title('LPD')
axs2[0].set_title('HPD')
axs3[0].set_title('HBD')
axs4[0].set_title('HSBD')

legend_handles1 = [
    axs0[0].scatter([],[],marker='s',color=colors[0],label=r'WFLO',edgecolor='k'),
    axs0[0].scatter([],[],marker='s',color=colors[1],label=r'WFFC',edgecolor='k'),
    axs0[0].scatter([],[],marker='s',color=colors[2],label=r'WFCO',edgecolor='k'),
]
leg1 = axs0[0].legend(handles=legend_handles1,ncols=3,loc='lower left', bbox_to_anchor=(1,1.15),title='Approach')
axs0[0].add_artist(leg1)
legend_handles2 = [
    axs4[0].plot([],[],linestyle='solid',color='k',linewidth=1,alpha=0.5,label='WFLO')[0],
    axs4[0].plot([],[],linestyle='dashed',color='k',linewidth=1,alpha=0.5,label='WFFC')[0],
    axs4[0].plot([],[],linestyle='dotted',color='k',linewidth=1,alpha=0.5,label='WFCO')[0],
]
leg2 = axs4[0].legend(handles=legend_handles2,title='Best layout',loc='lower right',ncol=3,bbox_to_anchor=(0,1.15))

axs0[0].set_ylabel(r'$\mathrm{COE_{€}}$ $[\mathrm{EUR}\, /\, \mathrm{MWh}]$')
axs0[1].set_ylabel(r'$\mathrm{COE_{CO2}}$ $[\mathrm{kgCO_2eq}\, /\, \mathrm{MWh}]$')

ax40_right = axs4[0].twinx()
ax41_right = axs4[1].twinx()
ax40_right.set_ylabel(r'$\Delta \,\mathrm{COE_{€}}$ $[\%]$',labelpad=40)
ax41_right.set_ylabel(r'$\Delta \,\mathrm{COE_{CO2}}$ $[\%]$',labelpad=40)
ax40_right.set_yticks([])
ax41_right.set_yticks([])

if savefig: plt.savefig(pathfig+'fig14'+'.'+formatfig,format=formatfig,bbox_inches='tight')
plt.show()


save_figure_as_pkl = False
load_figure_as_pkl = False

# save figure as pkl
if save_figure_as_pkl:
    with open('./fig14.pkl','wb') as f: 
        pickle.dump(fig,f)

# load figure as pkl
if load_figure_as_pkl:
    with open('./fig14.pkl','rb') as f: 
        fig = pickle.load(f)



#%%
# figure: scatter plot - case study: BL (AEP, AEPnet, LCOE, IOE)

savefig = False
formatfig = 'pdf'
pathfig = './'

# scatter plot (LCOE vs IOE)

colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887','#4dc064']

alpha = 0.4
s_all = 50
s_best = 150
marker_WFLO = 'o'
marker_WFFC = 'v'
marker_WFCO = 'X'
color_MO = colors[5]
color_AEP = colors[1]
color_AEPnet = colors[2]

fig,ax = plt.subplots(figsize=(6,5))

# layout optimization
ax.scatter(ds_HKNs_MO['LCOE'],ds_HKNs_MO['IOE'],c=color_MO,marker=marker_WFLO,alpha=alpha,s=s_all,zorder=-1)
ax.scatter(ds_HKNs_AEP['LCOE'],ds_HKNs_AEP['IOE'],c=color_AEP,marker=marker_WFLO,alpha=alpha,s=s_all,zorder=-1)
ax.scatter(ds_HKNs_AEPnet['LCOE'],ds_HKNs_AEPnet['IOE'],c=color_AEPnet,marker=marker_WFLO,alpha=alpha,s=s_all,zorder=-1)
ind_best_IOE = np.array(ds_HKNs_MO['IOE']).argmin()
ind_best_LCOE = np.array(ds_HKNs_MO['LCOE']).argmin()
ind_best_AEP = np.array(ds_HKNs_AEP['AEP']).argmax()
ind_best_AEPnet = np.array(ds_HKNs_AEPnet['AEPnet']).argmax()
ax.scatter(ds_HKNs_MO['LCOE'][ind_best_LCOE],ds_HKNs_MO['IOE'][ind_best_LCOE],c=color_MO,marker=marker_WFLO,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_MO['LCOE'][ind_best_IOE],ds_HKNs_MO['IOE'][ind_best_IOE],c=color_MO,marker=marker_WFLO,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_AEP['LCOE'][ind_best_AEP],ds_HKNs_AEP['IOE'][ind_best_AEP],c=color_AEP,marker=marker_WFLO,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_AEPnet['LCOE'][ind_best_AEPnet],ds_HKNs_AEPnet['IOE'][ind_best_AEPnet],c=color_AEPnet,marker=marker_WFLO,edgecolors='k',s=s_best)

# control optimization (GY)
ax.scatter(ds_HKNs_MO['LCOE_OY'],ds_HKNs_MO['IOE_OY'],c=color_MO,marker=marker_WFFC,alpha=alpha,s=s_all,zorder=-1)
ax.scatter(ds_HKNs_AEP['LCOE_OY'],ds_HKNs_AEP['IOE_OY'],c=color_AEP,marker=marker_WFFC,alpha=alpha,s=s_all,zorder=-1)
ax.scatter(ds_HKNs_AEPnet['LCOE_OY'],ds_HKNs_AEPnet['IOE_OY'],c=color_AEPnet,marker=marker_WFFC,alpha=alpha,s=s_all,zorder=-1)
ind_best_IOE = np.array(ds_HKNs_MO['IOE_OY']).argmin()
ind_best_LCOE = np.array(ds_HKNs_MO['LCOE_OY']).argmin()
ind_best_AEP = np.array(ds_HKNs_AEP['AEP_OY']).argmax()
ind_best_AEPnet = np.array(ds_HKNs_AEPnet['AEPnet_OY']).argmax()
ax.scatter(ds_HKNs_MO['LCOE_OY'][ind_best_LCOE],ds_HKNs_MO['IOE_OY'][ind_best_LCOE],c=color_MO,marker=marker_WFFC,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_MO['LCOE_OY'][ind_best_IOE],ds_HKNs_MO['IOE_OY'][ind_best_IOE],c=color_MO,marker=marker_WFFC,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_AEP['LCOE_OY'][ind_best_AEP],ds_HKNs_AEP['IOE_OY'][ind_best_AEP],c=color_AEP,marker=marker_WFFC,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_AEPnet['LCOE_OY'][ind_best_AEPnet],ds_HKNs_AEPnet['IOE_OY'][ind_best_AEPnet],c=color_AEPnet,marker=marker_WFFC,edgecolors='k',s=s_best)

# co-design optimization (GY)
ax.scatter(ds_HKNs_MO_GY['LCOE_OY'],ds_HKNs_MO_GY['IOE_OY'],c=color_MO,marker=marker_WFCO,alpha=alpha,s=s_all,zorder=-1)
ax.scatter(ds_HKNs_AEP_GY['LCOE_OY'],ds_HKNs_AEP_GY['IOE_OY'],c=color_AEP,marker=marker_WFCO,alpha=alpha,s=s_all,zorder=-1)
ax.scatter(ds_HKNs_AEPnet_GY['LCOE_OY'],ds_HKNs_AEPnet_GY['IOE_OY'],c=color_AEPnet,marker=marker_WFCO,alpha=alpha,s=s_all,zorder=-1)
ind_best_IOE = np.array(ds_HKNs_MO_GY['IOE_OY']).argmin()
ind_best_LCOE = np.array(ds_HKNs_MO_GY['LCOE_OY']).argmin()
ind_best_AEP = np.array(ds_HKNs_AEP_GY['AEP_OY']).argmax()
ind_best_AEPnet = np.array(ds_HKNs_AEPnet_GY['AEPnet_OY']).argmax()
ax.scatter(ds_HKNs_MO_GY['LCOE_OY'][ind_best_LCOE],ds_HKNs_MO_GY['IOE_OY'][ind_best_LCOE],c=color_MO,marker=marker_WFCO,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_MO_GY['LCOE_OY'][ind_best_IOE],ds_HKNs_MO_GY['IOE_OY'][ind_best_IOE],c=color_MO,marker=marker_WFCO,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_AEP_GY['LCOE_OY'][ind_best_AEP],ds_HKNs_AEP_GY['IOE_OY'][ind_best_AEP],c=color_AEP,marker=marker_WFCO,edgecolors='k',s=s_best)
ax.scatter(ds_HKNs_AEPnet_GY['LCOE_OY'][ind_best_AEPnet],ds_HKNs_AEPnet_GY['IOE_OY'][ind_best_AEPnet],c=color_AEPnet,marker=marker_WFCO,edgecolors='k',s=s_best)


# add axis in percentages
ind_best_AEP = np.array(ds_HKNs_AEP['AEP']).argmax()
ref_x = ds_HKNs_AEP['LCOE'][ind_best_AEP].values
ref_y = ds_HKNs_AEP['IOE'][ind_best_AEP].values
def x_to_pct(x): return 100*((x-ref_x)/ref_x)
def pct_to_x(p): return ((p+ref_x)/100)*ref_x
def y_to_pct(y): return 100*((y-ref_y)/ref_y)
def pct_to_y(p): return ((p+ref_y)/100)*ref_y
secax_x = ax.secondary_xaxis('top', functions=(x_to_pct, pct_to_x))
secax_y = ax.secondary_yaxis('right', functions=(y_to_pct, pct_to_y))
ax.axvline(ref_x, color='k', linestyle='--', linewidth=1,zorder=-2,alpha=0.5)
ax.axhline(ref_y, color='k', linestyle='--', linewidth=1,zorder=-2,alpha=0.5)

# legend
legend_handles1 = [
    Line2D([0],[0],marker='o',linestyle='',markerfacecolor='grey',markeredgecolor='grey',label='WFLO'),
    Line2D([0],[0],marker='v',linestyle='',markerfacecolor='grey',markeredgecolor='grey',label='WFFC'),
    Line2D([0],[0],marker='X',linestyle='',markerfacecolor='grey',markeredgecolor='grey',label='WFCO'),
]
leg1 = ax.legend(handles=legend_handles1,title='Approach:',loc='upper right',ncol=1,bbox_to_anchor=(1.65,0.75))
ax.add_artist(leg1)
legend_handles2 = [
    Line2D([0],[0],marker='s',linestyle='',markerfacecolor=colors[5],markeredgecolor='none',label=r'$\mathrm{COE_{EUR}}$ and $\mathrm{COE_{CO2}}$'),
    Line2D([0],[0],marker='s',linestyle='',markerfacecolor=colors[1],markeredgecolor='none',label=r'$\mathrm{AEP}$'),
    Line2D([0],[0],marker='s',linestyle='',markerfacecolor=colors[2],markeredgecolor='none',label=r'$\mathrm{AEPnet}$'),
]
leg2 = ax.legend(handles=legend_handles2,title='Objective:',loc='upper right',ncol=1,bbox_to_anchor=(1.65,1.))

fig.canvas.draw()

for leg in [leg1, leg2]:
    width=140
    leg._legend_box.set_width(width)
    leg._legend_box.align = "left"


ax.set_xlabel(r'$\mathrm{COE_{€}}$ $[\mathrm{EUR}\, /\, \mathrm{MWh}]$')
ax.set_ylabel(r'$\mathrm{COE_{CO2}}$ $[\mathrm{kgCO_2eq}\, /\, \mathrm{MWh}]$')
secax_x.set_xlabel(r'$\Delta\,\mathrm{COE_{€}}$ $[\%]$')
secax_y.set_ylabel(r'$\Delta\,\mathrm{COE_{CO2}}$ $[\%]$')
if savefig: plt.savefig(pathfig+'fig9'+'.'+formatfig,format=formatfig,bbox_inches='tight')

plt.show()


save_figure_as_pkl = False
load_figure_as_pkl = False

# save figure as pkl
if save_figure_as_pkl:
    with open('./fig9.pkl','wb') as f: 
        pickle.dump(fig,f)

# load figure as pkl
if load_figure_as_pkl:
    with open('./fig9.pkl','rb') as f: 
        fig = pickle.load(f)


#%%
# figure: scatter plot - case study: ALL (LCOE, IOE) - only layout optimization

savefig = False
formatfig = 'pdf'
pathfig = './'


# scatter plot (LCOE vs IOE)

colors = ['#001221','#c31c00','#1a6770','#144288','#a1227f']


fig,ax = plt.subplots(figsize=(5,4))

ax.scatter(ds_HKNs_MO['LCOE'],ds_HKNs_MO['IOE'],c=colors[0],marker='o',label='BL',s=20,alpha=0.4,zorder=0)
ax.scatter(ds_HKNs_MO['LCOE'][np.array(ds_HKNs_MO['LCOE']).argmin()],ds_HKNs_MO['IOE'][np.array(ds_HKNs_MO['LCOE']).argmin()],c=colors[0],marker='o',edgecolor='black',s=100,zorder=1)
ax.scatter(ds_HKNs_MO['LCOE'][np.array(ds_HKNs_MO['IOE']).argmin()],ds_HKNs_MO['IOE'][np.array(ds_HKNs_MO['IOE']).argmin()],c=colors[0],marker='o',edgecolor='black',s=100,zorder=1)

ax.scatter(ds_HKNsLPD_MO['LCOE'],ds_HKNsLPD_MO['IOE'],c=colors[1],marker='o',label='LPD',s=20,alpha=0.4,zorder=0)
ax.scatter(ds_HKNsLPD_MO['LCOE'][np.array(ds_HKNsLPD_MO['LCOE']).argmin()],ds_HKNsLPD_MO['IOE'][np.array(ds_HKNsLPD_MO['LCOE']).argmin()],c=colors[1],marker='o',edgecolor='black',s=100,zorder=1)
ax.scatter(ds_HKNsLPD_MO['LCOE'][np.array(ds_HKNsLPD_MO['IOE']).argmin()],ds_HKNsLPD_MO['IOE'][np.array(ds_HKNsLPD_MO['IOE']).argmin()],c=colors[1],marker='o',edgecolor='black',s=100,zorder=1)

ax.scatter(ds_HKNsHPD_MO['LCOE'],ds_HKNsHPD_MO['IOE'],c=colors[2],marker='o',label='HPD',s=20,alpha=0.4,zorder=0)
ax.scatter(ds_HKNsHPD_MO['LCOE'][np.array(ds_HKNsHPD_MO['LCOE']).argmin()],ds_HKNsHPD_MO['IOE'][np.array(ds_HKNsHPD_MO['LCOE']).argmin()],c=colors[2],marker='o',edgecolor='black',s=100,zorder=1)
ax.scatter(ds_HKNsHPD_MO['LCOE'][np.array(ds_HKNsHPD_MO['IOE']).argmin()],ds_HKNsHPD_MO['IOE'][np.array(ds_HKNsHPD_MO['IOE']).argmin()],c=colors[2],marker='o',edgecolor='black',s=100,zorder=1)

ax.scatter(ds_HKNsHBD_MO['LCOE'],ds_HKNsHBD_MO['IOE'],c=colors[3],marker='o',label='HBD',s=20,alpha=0.4,zorder=0)
ax.scatter(ds_HKNsHBD_MO['LCOE'][np.array(ds_HKNsHBD_MO['LCOE']).argmin()],ds_HKNsHBD_MO['IOE'][np.array(ds_HKNsHBD_MO['LCOE']).argmin()],c=colors[3],marker='o',edgecolor='black',s=100,zorder=1)
ax.scatter(ds_HKNsHBD_MO['LCOE'][np.array(ds_HKNsHBD_MO['IOE']).argmin()],ds_HKNsHBD_MO['IOE'][np.array(ds_HKNsHBD_MO['IOE']).argmin()],c=colors[3],marker='o',edgecolor='black',s=100,zorder=1)

ax.scatter(ds_HKNsHSBD_MO['LCOE'],ds_HKNsHSBD_MO['IOE'],c=colors[4],marker='o',label='HSBD',s=20,alpha=0.4,zorder=0)
ax.scatter(ds_HKNsHSBD_MO['LCOE'][np.array(ds_HKNsHSBD_MO['LCOE']).argmin()],ds_HKNsHSBD_MO['IOE'][np.array(ds_HKNsHSBD_MO['LCOE']).argmin()],c=colors[4],marker='o',edgecolor='black',s=100,zorder=1)
ax.scatter(ds_HKNsHSBD_MO['LCOE'][np.array(ds_HKNsHSBD_MO['IOE']).argmin()],ds_HKNsHSBD_MO['IOE'][np.array(ds_HKNsHSBD_MO['IOE']).argmin()],c=colors[4],marker='o',edgecolor='black',s=100,zorder=1)

legend_handles1 = [
    Line2D([0],[0],marker='.',markersize=20,linestyle='',markerfacecolor=colors[0],markeredgecolor='none',label=r'BL'),
    Line2D([0],[0],marker='.',markersize=20,linestyle='',markerfacecolor=colors[1],markeredgecolor='none',label=r'LPD'),
    Line2D([0],[0],marker='.',markersize=20,linestyle='',markerfacecolor=colors[2],markeredgecolor='none',label=r'HPD'),
    Line2D([0],[0],marker='.',markersize=20,linestyle='',markerfacecolor=colors[3],markeredgecolor='none',label=r'HBD'),
    Line2D([0],[0],marker='.',markersize=20,linestyle='',markerfacecolor=colors[4],markeredgecolor='none',label=r'HSBD'),
]
leg1 = ax.legend(handles=legend_handles1,title='Case study',loc='upper left',ncol=1)
ax.add_artist(leg1)

ax.set_xlabel(r'$\mathrm{COE_{€}}$ $[\mathrm{EUR}\, /\, \mathrm{MWh}]$')
ax.set_ylabel(r'$\mathrm{COE_{CO2}}$ $[\mathrm{kgCO_2eq}\, /\, \mathrm{MWh}]$')

if savefig: plt.savefig(pathfig+'fig13'+'.'+formatfig,format=formatfig,bbox_inches='tight')
plt.show()


save_figure_as_pkl = False
load_figure_as_pkl = False

# save figure as pkl
if save_figure_as_pkl:
    with open('./fig13.pkl','wb') as f: 
        pickle.dump(fig,f)

# load figure as pkl
if load_figure_as_pkl:
    with open('./fig13.pkl','rb') as f: 
        fig = pickle.load(f)



# %%
