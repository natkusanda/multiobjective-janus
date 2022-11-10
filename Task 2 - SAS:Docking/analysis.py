import os, sys, glob
import pandas as pd
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
from rdkit.Chem import Draw
RDLogger.DisableLog("rdApp.*")

# LogP, QED import
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import qed

import olympus
from olympus.utils.misc import get_hypervolume, get_pareto, get_pareto_set
# from janus_olympus import create_value_space, create_scalarizer, scalarize_and_sort
# import plotly.express as px
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from cairosvg import svg2png



def dist_to_utopia(fitness, utopia, objective='min'):
    # get the fitness to utopia
    fitness = np.array(fitness, dtype=float)
    utopia = np.stack([np.array(utopia, dtype=float)]*fitness.shape[0], axis=0)

    assert fitness.shape == utopia.shape, 'Utopia and fitness does not match in shape'

    if objective == 'max':
        x = utopia - fitness
    elif objective == 'min':
        x = fitness - utopia
    else:
        raise ValueError('Pick max or min for objective')
    dist = (np.maximum(0, x)/np.abs(utopia))**2
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)

    return dist

# def get_trace(df, column, objective='min'):
#     # get the trace info for `column` of `df` dataframe
#     if objective == 'max':
#         pass
#     elif objective == 'min':
#         pass
#     else:
#         raise ValueError('Pick max or min for objective')
    
#     return

def read_fitness(filename, n_fitness):
    with open(filename, 'r') as f:
        content = f.read()
    content = content.strip().replace('[','').replace(']', '').split()
    fitness = np.array(content, dtype=float).reshape((-1, n_fitness), order='c')
    return fitness

def read_smiles(filename):
    with open(filename, 'r') as f:
        content = f.read()
    smiles = content.strip().split()
    # fitness = np.array(content, dtype=float).reshape((-1, n_fitness), order='c')
    return np.array(smiles)


if __name__ == '__main__':
    # specify informations about runs
    run_types = ['ctrl', 'random', 'chimera', 'hv']
    # run_types = ['random', 'chimera', 'hv']
    n_gen = 50          # number of generations
    n_run = 10          # number of repeated runs
    n_fitness = 2       # number of fitness dimensions (# of objectives)
    need_smiles = True # set to True if you want to access the smiles (takes longer)
    utopia = np.array([1.0, -20.0])

    # loop through all files
    all_df = []
    trace_df = []
    for run in run_types:
        for i in tqdm(range(n_run), desc=f'Currently reading {run}'):

            best_dtu = np.inf
            trace = []
            for j in tqdm(range(n_gen), leave=False):
                if need_smiles:
                    local_fit = read_fitness(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/fitness_local_search.txt', n_fitness)
                    explore_fit = read_fitness(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/fitness_explore.txt', n_fitness)
                    local_smi = read_smiles(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/population_local_search.txt')
                    explore_smi = read_smiles(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/population_explore.txt')
                    fitness = np.append(explore_fit, local_fit, axis=0)
                    smiles = np.append(explore_smi, local_smi)
                else:
                    fitness = read_fitness(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/final_gen_fitness.txt', n_fitness)

                dtu = dist_to_utopia(fitness, utopia, objective='min')

                # store best dtu
                if np.min(dtu) < best_dtu:
                    best_dtu = np.min(dtu)
                trace.append(best_dtu)

                res = {
                    'run_type': [run]*len(dtu),
                    'run_id': [i]*len(dtu),
                    'gen': [j+1]*len(dtu),
                    'fitness': fitness.tolist(),
                    'dist_to_utopia': dtu
                }
                if need_smiles:
                    res.update({'smiles': smiles})

                all_df.append(pd.DataFrame(res))

            trace_df.append(pd.DataFrame({
                'run_type': [run]*len(trace),
                'run_id': [i]*len(trace),
                'trace': trace
            }))

    trace_df = pd.concat(trace_df)
    all_df = pd.concat(all_df)

    # plot the traces
    # rename for plotting
    trace_df = trace_df.reset_index().rename(columns={'index': 'Generation', 'trace': 'Minimum distance'})
    trace_df['Generation'] = trace_df['Generation'] + 1
    trace_df['run_type'] = trace_df['run_type'].map({
        'chimera': 'Chimera', 
        'hv': 'Hypervolume', 
        'ctrl': 'WeightedSum', 
        'random': 'Random'
    })

    fig, ax = plt.subplots(figsize=(10,8))
    g = sns.lineplot(data=trace_df, x='Generation', y='Minimum distance', hue='run_type', errorbar=("sd", 1), ax=ax)
    g.get_legend().set_title(None)
    ax.set_xlim([0, n_gen])
    ax.set_ylabel('Minimum Distance', fontsize=25)
    ax.set_xlabel('Generation', fontsize=25)
    ax.grid(True, linestyle='-.')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(prop={'size': 25})

    plt.savefig('dtu_sas-docking.png', bbox_inches='tight')
    plt.close()

    # Plot utopia scatter plot
    # read zinc pareto front
    zinc_df = pd.read_csv('pareto_tagged_sas_docking.csv')
    zinc_df = zinc_df[zinc_df['pareto'] == 1].sort_values('SAS')
    zinc_df = zinc_df[zinc_df['SAS'] < 2.5]
    pareto_front = zinc_df[['SAS', '1oyt']].to_numpy()
    map_names = {
        'chimera': 'Chimera', 
        'hv': 'Hypervolume', 
        'ctrl': 'WeightedSum', 
        'random': 'Random'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,0.50]}, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05) 
    for run in run_types:
        points = np.array(
            all_df[all_df['run_type'] == run].drop_duplicates('dist_to_utopia').nsmallest(30, 'dist_to_utopia')['fitness'].tolist(), 
            dtype=float
        )
        ax1.scatter(points[:, 0], points[:, 1], label=map_names[run], edgecolors='white', s=60)

    ax2.scatter(utopia[0], utopia[1], label='Utopian point', marker='X', c='#7f7f7f')
    ax1.plot(pareto_front[:,0], pareto_front[:,1], marker='s', label='Zinc', c='#9e9ac8')

    # ax2.set_ylim([-18.5, -21.5])
    ax2.set_yticks([-20.0])
    ax1.tick_params(axis='x', which='both', top=False, bottom=False)

    # merge two figures
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.xaxis.tick_bottom()
    ax1.tick_params(axis='both', which='both', labelsize=20)
    ax2.tick_params(axis='both', which='both', labelsize=20)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.set_ylabel('Docking Score (kcal/mol)', fontsize=20)
    ax1.yaxis.set_label_coords(0.02, 0.5, transform=fig.transFigure)
    ax2.set_xlabel('SAS', fontsize=20)

    # combined legend
    handles, labels = ax1.get_legend_handles_labels()
    h, l = ax2.get_legend_handles_labels()
    handles.insert(-1, h[0])
    labels.insert(-1, l[0])
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.88), prop={'size': 20})

    handles[-1]._markersize = 10.0
    h[0]._sizes=[500]

    plt.savefig('dtu_scatter.png', bbox_inches='tight')
    
    # plot the hypervolumes

    # sns.scatterplot(data=df_scatter, x='S)

    # plot the best smiles
    if need_smiles:
        for run in run_types:
            best_10 = all_df[all_df['run_type'] == run].drop_duplicates('dist_to_utopia').nsmallest(10, 'dist_to_utopia')
            mols = [Chem.MolFromSmiles(smi) for smi in best_10['smiles']]
            labs = [f'SAS: {sas:.3f}\n\n\n\n\nDocking:{dock}' for sas, dock in best_10['fitness']]

            n=320
            d2d = Draw.MolDraw2DSVG(n*5, n*2, n, n)
            d2d.drawOptions().addStereoAnnotation=True
            d2d.drawOptions().legendFontSize = 90
            d2d.drawOptions().legendFraction = 0.2
            d2d.DrawMolecules(mols,legends=labs)
            d2d.FinishDrawing()
            svg_string = d2d.GetDrawingText()

            svg2png(bytestring=svg_string, write_to=f'molecules_{run}.png')

            

            # img = Draw.MolsToGridImage(mols, legends=labs, molsPerRow=5, subImgSize=(200,200))
    # import pdb; pdb.set_trace()
    #     n=300
    #     d2d = Draw.MolDraw2DSVG(n*5, n*2, n, n)
    #     d2d.drawOptions().addStereoAnnotation=True
    #     d2d.drawOptions().legendFontSize = 100
    #     d2d.drawOptions().legendFraction = 0.2
    #     d2d.DrawMolecules(mols,legends=labs)
    #     d2d.FinishDrawing()
    #     svg_string = d2d.GetDrawingText()
        # svg2png(bytestring=svg_string, write_to=f'molecules_{run}.png')


            # img.save(f'molecules_{run}.png')


    # import pdb; pdb.set_trace()