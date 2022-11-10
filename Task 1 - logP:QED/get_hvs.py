
# %%
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import numpy as np

import olympus
from olympus.utils.misc import get_hypervolume, get_pareto, get_pareto_set
import plotly.express as px
import plotly.graph_objects as go
# %%
def compute_pareto(optim_type):
    logp_worst = -6.8762
    qed_worst = 0.025882

    df = pd.read_csv("smiles_collector_" + optim_type + ".csv")

    smiles = list(df['smi'])
    logp = list(df['logp'])
    qed = list(df['qed'])
    idx = range(len(smiles))
    params = np.array([idx]).T

    # 2) logp, qed
    data = np.column_stack((logp,qed))
    # Negative for maximization
    data[:,0] = - data[:,0]
    data[:,1] = - data[:,1]
    
    pareto_front, pareto_set = get_pareto_set(params, data)
    #df['pareto'] = [1 if i in pareto_set else 0.5 for i in range(len(idx))]
    #df.to_csv(optim_type + "_pareto_tagged_logp_qed.csv")
    print("logp, qed Done!")

    # Negative for maximization
    w_ref = np.array([-logp_worst,-qed_worst])
    hypervolume_1 = get_hypervolume(pareto_front, w_ref)

    return hypervolume_1
# %%
# %%
if __name__ == "__main__":

    hvs = []
    optim_types = ['ctrl','random','chimera','hv']
    for optim_type in optim_types:
        hvs_type = [] 
        print(optim_type)
        hv = compute_pareto(optim_type)
        hvs_type.append(optim_type)
        hvs_type.append(hv)

        for i in range(2,11):
            optim_string = optim_type + str(i)
            hv = compute_pareto(optim_string)
            hvs_type.append(hv)
        hvs.append((hvs_type))
    
    df_hvs = pd.DataFrame(hvs, columns =['optim_type','1','2','3','4','5','6','7','8','9','10'])
    df_hvs.to_csv("all_hvs.csv")




# %%


