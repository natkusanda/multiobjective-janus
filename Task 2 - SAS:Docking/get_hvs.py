
# %%
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import olympus
from olympus.utils.misc import get_hypervolume, get_pareto, get_pareto_set
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from tqdm import tqdm

def compute_pareto(optim_type):
    df = pd.read_csv("smiles_collector_" + optim_type + ".csv")
    # df = pd.read_csv("ZINC_1oyt.csv")
    # df = df.iloc[:1000]
    df = df.rename(columns={'smiles': 'smi'})
    df = df[df['1oyt'] != 999.0]    # drop the failed docking

    # logp_worst = -6.8762
    # qed_worst = 0.025882
    sas_worst = df['SAS'].max()
    docking_worst = df['1oyt'].max() 

    smiles = list(df['smi'])
    # logp = list(df['logp'])
    # qed = list(df['qed'])
    docking = list(df['1oyt'])
    sas = list(df['SAS'])
    idx = range(len(smiles))
    params = np.array([idx]).T

    # 2) logp, qed
    # data = np.column_stack((logp,qed))
    data = np.column_stack((docking, sas))
    # Negative for maximization
    # data[:,0] = - data[:,0]
    # data[:,1] = - data[:,1]
    
    pareto_front, pareto_set = get_pareto_set(params, data)
    df['pareto'] = [1 if i in pareto_set else 0.5 for i in range(len(idx))]
    # df.to_csv(optim_type + "_pareto_tagged_logp_qed.csv")
    df.to_csv("pareto_tagged_sas_docking.csv", index=False)
    print("logp, qed Done!")

    # Negative for maximization
    w_ref = np.array([docking_worst, sas_worst])
    hypervolume_1 = get_hypervolume(pareto_front, w_ref)

    return hypervolume_1
# %%
# %%
def read_fitness(filename, n_fitness):
    with open(filename, 'r') as f:
        content = f.read()
    content = content.strip().replace('[','').replace(']', '').split()
    fitness = np.array(content, dtype=float).reshape((-1, n_fitness), order='c')
    return fitness

def get_hv(fitness, ref):
    idx = range(len(fitness))
    params = np.array([idx]).T
    pareto_front, pareto_set = get_pareto_set(params, fitness) 

    hv = get_hypervolume(pareto_front, ref)
    return hv, pareto_front

def get_pareto_info(fitness):
    idx = range(len(fitness))
    params = np.array([idx]).T
    pareto_front, pareto_set = get_pareto_set(params, fitness) 
    return pareto_front

def compare_pareto(desired, reference):
    # assumes minimization
    # get the first and last of desired and reference
    f_des, l_des = desired[desired[:,0].argmax(), :], desired[desired[:, 1].argmax(), :]
    # f_ref, l_ref = reference[reference[:,0].argmax(), 1], reference[reference[:, 1].argmax(), 1]
    # reference[(reference[:,0] > f_des[0]) & (reference[:,1] < f_des[1])] 

    new_pareto = [
        desired,
        reference[(reference[:,0] > f_des[0]) & (reference[:,1] < f_des[1])], 
        reference[(reference[:,0] < l_des[0]) & (reference[:,1] > l_des[1])] 
    ]
    new_pareto = np.concatenate(new_pareto, axis=0)

    return new_pareto


if __name__ == "__main__":

    # get the reference point
    df = pd.read_csv('pareto_tagged_sas_docking.csv')
    df = df[df['1oyt'] != 999.0]
    # sns.scatterplot(data=df, x = 'SAS', y='1oyt', hue='pareto')
    # plt.savefig('docking_pareto.png', bbox_inches='tight')
    
    hvs = []
    optim_types = ['ctrl', 'random','chimera','hv'] # ['ctrl','random','chimera','hv']

    n_gen = 50      # number of generations
    n_run = 10      # number of repeated runs
    n_fitness = 2   # number of fitness dimensions (# of objectives)
    threshold = 10.0
    
    # from previous run, worst point is:
    # w_ref = np.array([7.43576036, 152.4])
    # worst in zinc_hv
    w_ref = np.array([7.2234205, 3.9])
    # import pdb; pdb.set_trace()

    # caluclate for zinc
    df_pareto = df[df['pareto'] == 1.0]
    df_pareto = df_pareto[df_pareto['SAS'] < threshold]  # only look at points below 2.0
    zinc_pareto = df_pareto[['SAS', '1oyt']].to_numpy()
    zinc_hv = get_hypervolume(zinc_pareto, w_ref)
    print(zinc_hv)

    map_names = {
        'chimera': 'Chimera', 
        'hv': 'Hypervolume', 
        'ctrl': 'WeightedSum', 
        'random': 'Random'
    }

    all_df = []
    res = {'Approach':[], 'run_id': [], 'Hypervolume': [], 'ref': []}
    for run in optim_types:
        for i in tqdm(range(n_run), desc=f'Currently running {run}'):
            # fig, ax = plt.subplots()
            all_fitness = []
            for j in tqdm(range(n_gen), leave=False):
                local_fit = read_fitness(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/fitness_local_search.txt', n_fitness)
                explore_fit = read_fitness(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/fitness_explore.txt', n_fitness)
                # local_smi = read_smiles(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/population_local_search.txt')
                # explore_smi = read_smiles(f'{i}_ai4mat/janus_{run}/RESULTS/{j}_DATA/population_explore.txt')

                fitness = np.append(explore_fit, local_fit, axis=0)
                all_fitness.append(fitness)

            # all_fitness.append(df_pareto[['SAS', '1oyt']].to_numpy())
            all_fitness = np.concatenate(all_fitness, axis=0)
            all_fitness = all_fitness[all_fitness[:,0] < threshold]
            # print(all_fitness.shape)

            # import pdb; pdb.set_trace()
            pareto_front = get_pareto_info(all_fitness)
            # pareto_front = compare_pareto(pareto_front, zinc_pareto)
            hv = get_hypervolume(pareto_front, w_ref)


            # ax.scatter(pareto_front[:,0], pareto_front[:,1], label='run')
            # ax.scatter(zinc_pareto[:,0], zinc_pareto[:,1], label='ZINC')

            # plt.savefig(f'pareto_tests/{run}_{i}_pareto.png')
            # plt.close()
            
            # n = fitness.shape[0]
            res['Approach'].append(map_names[run])
            res['run_id'].append(i)
            res['Hypervolume'].append(hv)
            res['ref'].append(w_ref)
            # all_df.append(pd.DataFrame(res))

    # all_df = pd.concat(all_df)
    all_df = pd.DataFrame(res)
    # import pdb; pdb.set_trace()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.violinplot(data=all_df, x='Approach', y='Hypervolume', ax= ax)
    # plt.axhline(y=1065.086811, label='ZINC_red average', linestyle='--')
    # plt.axhline(y=123.9069, label='ZINC_red average', linestyle='--')
    plt.axhline(y=zinc_hv, label='ZINC_red average', linestyle='--', c='k')
    ax.annotate('ZINC_red',xy=(1.0,zinc_hv - 2.0),fontsize=20)
    plt.ylabel('Hypervolume', fontsize=20)
    plt.xlabel('Approach', fontsize=20)
    plt.xticks(rotation=15)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig('hv_violin.png', bbox_inches='tight')
    # import pdb; pdb.set_trace()

    # all_df['Hypervolume'].groupby('Approach').mean()

    # import pdb; pdb.set_trace()
    # worst_sas = df['SAS'].max()
    # worst_docking = df['1oyt'].max()
    # fitness = np.array(all_df['fitness'].tolist(), dtype=float)
    # if fitness[:,0].max() > worst_sas:
    #     print('Runs worst than ZINC for SAS')
    #     worst_sas = fitness[:,0].max()
    # if fitness[:,1].max() > worst_docking:
    #     print('Runs worst than ZINC for docking')
    #     worst_docking = fitness[:,1].max()
    # w_ref = np.array([worst_sas, worst_docking])

    
    
    # for run in optim_types:
    #     tmp_df = all_df[all_df['run_type'] == run]

    





                





####
    #             hvs_type = [] 
    #             print(optim_type)
    #             hv = compute_pareto(optim_type)
    #             hvs_type.append(optim_type)
    #             hvs_type.append(hv)

    #             for i in range(2,11):
    #                 optim_string = optim_type + str(i)
    #                 hv = compute_pareto(optim_string)
    #                 hvs_type.append(hv)
    #             hvs.append((hvs_type))
    
    # df_hvs = pd.DataFrame(hvs, columns =['optim_type','1','2','3','4','5','6','7','8','9','10'])
    # df_hvs.to_csv("all_hvs.csv")




# %%


