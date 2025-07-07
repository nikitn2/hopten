import os; os.environ["KMP_WARNINGS"] = "0"; os.environ["JAX_PLATFORM_NAME"] = "cpu"; os.environ["NUMBA_DISABLE_JIT"] = "1"
#import requests, zipfile, io
import numpy as np
import sys; sys.path.insert(1, '../tendeq/')
import pandas as pd
import config_hopten as cfg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Hack to deal with numpy mismatch between local and Oxford ARC clusters
import sys, types; sys.modules["numpy._core"] = types.ModuleType("numpy._core"); sys.modules["numpy._core.numeric"] = np.core.numeric; sys.modules["numpy._core.multiarray"] = np.core.multiarray

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

if __name__ == '__main__':
    
    # Select datasets & poly-orders
    datasets = ["years", "casp", "wines", "houses"]
    polys    = [1,2,3,4]
    
    # Load datasets
    dfs = []
    for dataset in datasets:
        for poly in polys:
            out_file = os.path.join(cfg.dir_data, "out", f"{dataset}_poly{poly}.pkl")
            if os.path.isfile(out_file): dfs.append(pd.read_pickle(out_file))
        
    # Concatenate into single dataframe
    df_full = pd.concat(dfs, axis=0)
    d = df_full.reset_index()                    # flatten MultiIndex if present

    # Plot
    # ---------- colour map : dataset ----------
    tab_colors = plt.cm.tab10.colors
    datasets   = sorted(d['dataset'].unique())
    colour_map = {ds: tab_colors[i % len(tab_colors)] for i, ds in enumerate(datasets)}

    # ---------- transparency : cutoff_X ----------
    log_cut = np.log10(d['cutoff_X'])
    alpha   = 1 - (log_cut - log_cut.min()) / (log_cut.max() - log_cut.min())
    alpha   = alpha.clip(0.1, 1.0)
    d['rgba'] = [
        tuple(list(colour_map[ds]) + [al])
        for ds, al in zip(d['dataset'], alpha)]
    
    
    # ---------- marker size: poly-order ----------
    # Normalise sizes between two pleasant limits (pts², because scatter‐s)
    poly_min, poly_max       = d['poly'].min(), d['poly'].max()
    size_min, size_max       = 20, 100                      # tweak if you like
    d['size'] = size_min + (d['poly'] - poly_min) / (poly_max - poly_min) * (size_max - size_min)
    
    # ---------- marker shapes : χ_MPO ----------
    shapes = {2: '^', 4: 's', 6: 'p', 8: 'o'}

    fig, ax = plt.subplots(figsize=(7, 7))
    for chi, mk in shapes.items():
     m = d['chi_mpo'] == chi
     if m.any():
         ax.scatter(
             d.loc[m, 'score_class'],
             d.loc[m, 'score_mpo'],
             marker   = mk,
             c        = d.loc[m, 'rgba'],
             s        = d.loc[m, 'size']      # ← new
         )

    # 45° reference
    lo, hi = [d[['score_class', 'score_mpo']].min().max(),
              d[['score_class', 'score_mpo']].max().max()]
    ax.plot([lo, hi], [lo, hi], ls='-', lw=1, color='grey')

    # labels
    ax.set_xlabel('Classical ridge  $R^{2}$')
    ax.set_ylabel('MPO ridge  $R^{2}$')
    ax.set_title('MPO vs. Classical ridge regression')

    # ---------- legend 1: dataset colours ----------
    ds_handles = [
        Line2D([], [], marker='o', linestyle='None',
               markerfacecolor=colour_map[ds], markeredgecolor='k',
               label=ds.capitalize(), markersize=9)
        for ds in datasets
    ]
    legend1 = ax.legend(handles=ds_handles, title='Dataset',
                        loc='lower left', frameon=True,
                        bbox_to_anchor=(0.01, 0.81))
    ax.add_artist(legend1)

    # ---------- legend 2: χ shapes ----------
    shape_handles = [
        Line2D([], [], marker=mk, linestyle='None',
               color='k', label=f'$\\chi$ = {chi}', markersize=9, alpha=0.5)
        for chi, mk in shapes.items()
    ]
    legend2 = ax.legend(handles=shape_handles, title='$\\chi_{\\mathrm{MPO}}$',
                        loc='lower left', frameon=True,
                        bbox_to_anchor=(0.01, 0.615))
    ax.add_artist(legend2)

    # ---------- legend 3: transparency text box ----------
    dummy_handle = Line2D([], [], linestyle='None')
    legend3 = ax.legend(handles=[dummy_handle],
                        labels=['More opaque →\nless data-compression;\nsvd-cutoff range =\n'+r"($10^{-16},10^{-12},10^{-8},10^{-4}$)"],
                        handlelength=0, loc='lower left', frameon=True,
                        bbox_to_anchor=(0.54, 0.37))
    ax.add_artist(legend3)
    
    # ---------- legend 4: poly-order text box ----------
    dummy_handle = Line2D([], [], linestyle='None')
    legend4 = ax.legend(handles=[dummy_handle],
                        labels=['Larger symbol→\nhigher poly-order;\npoly range = (1,2,3,4)'],
                        handlelength=0, loc='lower left', frameon=True,
                        bbox_to_anchor=(0.58, 0.27))

    
    # Plot and save
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    plt.tight_layout()
    
    out_png = os.path.join(cfg.dir_data, "mpo_vs_classical_r2.png")
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
