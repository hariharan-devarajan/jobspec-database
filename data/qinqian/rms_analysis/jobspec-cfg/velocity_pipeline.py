#!/usr/bin/env python
import sys
import os
#import stream as st
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import matplotlib.pyplot as plt
import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
scv.settings.verbosity = 3
scv.settings.set_figure_params('scvelo')

def compute_velocity(loom, mito_prefix='MT-', cutoff=[5000, 0.15], norm=False):
    if isinstance(loom, str):
#         adata = scv.read(loom, cache=True) ## this is not working for Seurat output loom file
        adata= sc.read_loom(loom)
    elif len(loom) == 1:
        adata = scv.read(loom[0], cache=True)
    elif len(loom) > 1:
        adata = scv.read(loom[0], cache=True)
        adata2 = scv.read(loom[1], cache=True)
        adata = adata.concatenate(adata2, batch_key='library')
    adata.var_names_make_unique()
    cells = adata.obs.index
    print(adata.layers)
    if not norm:
        sc.pp.filter_cells(adata, min_genes=1000)
        #sc.pp.filter_genes(adata, min_cells=5)
        mito_genes = adata.var_names.str.startswith(mito_prefix)
        # for each cell compute fraction of counts in mito genes vs. all genes
        # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
        adata.obs['percent_mito'] = np.sum(
            adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        # add the total counts per cell as observations-annotation to adata
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1    
        sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
                     jitter=0.4, multi_panel=True)  
    #     print(adata.obs['percent_mito'])
        adata = adata[adata.obs['n_genes'] < cutoff[0], :]
        adata = adata[adata.obs['percent_mito'] < cutoff[1], :]     
        
    scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=3000)    
    scv.pp.moments(adata, n_pcs=20, n_neighbors=30)
  #  sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
  #  sc.pp.scale(adata, max_value=10)
    if 'X_umap' not in adata.obsm.keys():
        sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution=0.8)
    scv.tl.velocity(adata, mode='stochastic')
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis='umap')
    scv.tl.rank_velocity_genes(adata, match_with='clusters', resolution=0.8)
    scv.tl.terminal_states(adata, groupby='velocity_clusters')
    scv.tl.velocity_pseudotime(adata, groupby='velocity_clusters')
    select = np.in1d(cells, adata.obs.index.values)
    return adata, select


def load_seurat_umap(args, test):
    test_seu = r('readRDS')(args.seurat)
    #meta = pandas2ri.ri2py(r['data.frame'](test_seu.slots['meta.data']))
    if args.species == 'human': 
        #clusters = meta.loc[test[1], 'RNA_snn_res.0.8']
        clusters = pandas2ri.ri2py(r['data.frame'](test_seu.slots['meta.data']).rx2('RNA_snn_res.0.8'))[test[1]]
    else:
        #clusters = meta.loc[test[1], 'seurat_clusters']
        clusters = pandas2ri.ri2py(r['data.frame'](test_seu.slots['meta.data']).rx2('seurat_clusters'))[test[1]]

    print(clusters)
    if args.species == 'human':
        reductions = pandas2ri.ri2py(r['data.frame'](r['slot'](test_seu.slots['reductions'].rx2('umap'), 'cell.embeddings')))
        test[0].obsm['X_umap']  = reductions.loc[test[1], :].values
        red = reductions.loc[test[1], :]
    else:   # fish used the spliced umap
        reductions = test[0].obsm['umap_cell_embeddings'] 
        test[0].obsm['X_umap']  = test[0].obsm['umap_cell_embeddings']  ## already filtered again by scvelo, no need [test[1],:]
        red = pd.DataFrame(reductions)

    print(pd.DataFrame(test[0].obsm['X_umap']).describe())
    #test[0].obs['clusters'] = clusters.values
    #test[0].obs['louvain']  = clusters.values
    test[0].obs['clusters'] = clusters
    test[0].obs['louvain']  = clusters

    sc.tl.paga(test[0])
    sc.tl.paga(test[0], groups='clusters', use_rna_velocity=False)
    pos = test[0].obsm['X_umap']
    pos_raw = []
    for i in sorted(clusters.unique()):
        redt = red.loc[(clusters==i), :]
        pos_raw.append(redt.median(axis=0).values)
    pos = np.vstack(pos_raw)
    if args.species == 'human':
        metalabels = np.array(["GROUND", "Hypoxia", "EMT", "G1S", "UNASSIGNED", "G2M", "MUSCLE", "INTERFERON", "PROLIF", "Histones"])
        metacolors = np.array(["#A6A6A6", "#F19545", "#672366", "#3465FC", "#F2F2F2",
                               "#3465FC", "#E93F33", "#418107", "#3465FC", "#FDF731"])
        colortab = pd.read_table('color_table.xls')
        test[0].uns['clusters_colors'] = metacolors[pd.match(colortab.loc[:, args.name].dropna(), metalabels)]
    else:
        colortab = pd.read_table('fish_color_table.txt')

    print(test[0].obs['clusters'].cat.categories)
    print(np.array([str(i) + '_' +colortab.loc[:, args.name].dropna()[i] for i in range(len(colortab.loc[:, args.name].dropna()))]))
    test[0].obs['clusters'].cat.categories = np.array([str(i) + '_' +colortab.loc[:, args.name].dropna()[i] for i in range(len(colortab.loc[:, args.name].dropna()))])
    print(test[0].obs['clusters'].cat.categories)
    return clusters, reductions, pos


def plot_velocity(test, name, pos):
    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(30, 20)
    ax1=scv.pl.velocity_embedding_stream(test[0], basis='umap', color=['clusters'], 
                                         legend_fontsize=15, show=False, ax=axes[0][0])

    ax2=scv.pl.scatter(test[0], color=['root_cells'], size=100, show=False, ax=axes[0][1], legend_fontsize=16)
    ax3=scv.pl.scatter(test[0], color=['end_points'], size=100, show=False, ax=axes[0][2], legend_fontsize=16)
    #print(test[0].obs.velocity_pseudotime)
    #ax4=scv.pl.scatter(test[0], color=['velocity_pseudotime'],  size=100, show=False, ax=axes[1][0], legend_fontsize=16)
    ax5=sc.pl.paga(test[0], color=['clusters'], edge_width_scale=0.5, layout='fr', random_state=0, frameon=False,
                   fontsize=20, threshold=0.2, ax=axes[1][1], pos=pos)

    ax6=scv.pl.velocity_graph(test[0], ax=axes[1][2], legend_fontsize=16)
    for a in axes:
        for ax in a:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                          ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(25)
    #plt.savefig(f'../results/{name}_velocity_paga.pdf')
    plt.savefig(f'../results/{name}_velocity_paga.png') ## due to ValueError: Can only output finite numbers in PDF


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loom',    default='test', help='loom file path')
    parser.add_argument('-s', '--seurat',  default='test', help='seurat object path')
    parser.add_argument('-n', '--name',    default='test', help='output file name')
    parser.add_argument('--species',       default='human', help='output file name')
    args = parser.parse_args()
    pandas2ri.activate()
    print(args)
    if args.loom == 'test' or args.seurat == 'test':
        parser.print_help()
        sys.exit(1)

    print(args.loom)
    if os.path.exists(f'../results/velocity_previousversion/{args.name}_velocity.h5ad') and os.path.exists(f'../results/velocity_previousversion/{args.name}_velocity.npy'):
        test = []
        test.append(sc.read_h5ad(f'../results/velocity_previousversion/{args.name}_velocity.h5ad'))
        test.append(np.load(f'../results/velocity_previousversion/{args.name}_velocity.npy'))
    else:
        if args.species == 'fish':
            test = compute_velocity(args.loom,
                                    mito_prefix='mt-', cutoff=[4000, 0.1], norm=False)
        else:
            test = compute_velocity(args.loom,
                                    mito_prefix='MT-', cutoff=[8000, 0.2], norm=False)
        test[0].write(f'../results/velocity_previousversion/{args.name}_velocity.h5ad')
        np.save(f'../results/velocity_previousversion/{args.name}_velocity.npy', test[1])

    clusters, reductions, pos = load_seurat_umap(args, test)
    print(pos)
    rank_genes = pd.DataFrame(test[0].uns['rank_velocity_genes']['names'])
    print(rank_genes.head())
    plot_velocity(test, args.name, pos)

