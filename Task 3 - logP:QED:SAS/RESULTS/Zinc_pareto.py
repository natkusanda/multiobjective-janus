
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
    qed_worst = 0.111811475018
    sas_worst = 7.289282840617412

    if optim_type == 'zinc':
        df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv")
        smiles = list(df['smiles'])
        logp = list(df['logP'])
        qed = list(df['qed'])
        sas = list(df['SAS'])

    else:
        df = pd.read_csv("RESULTS_" + optim_type + "/smiles_collector.csv")

        smiles = list(df['smi'])
        logp = list(df['logp'])
        qed = list(df['qed'])
        sas = list(df['sas'])
        
    idx = range(len(smiles))
    params = np.array([idx]).T

    # 2) logp, qed
    data = np.column_stack((logp,qed,sas))
    data[:,0] = - data[:,0]
    data[:,1] = - data[:,1]
    
    pareto_front, pareto_set = get_pareto_set(params, data)
    df['pareto'] = [1 if i in pareto_set else 0.5 for i in range(len(idx))]
    df.to_csv(optim_type + "_pareto_tagged_logp_qed.csv")
    print("logp, qed, sas Done!")

    w_ref = np.array([-logp_worst, -qed_worst, sas_worst])
    hypervolume_1 = get_hypervolume(pareto_front, w_ref)
    # get_r2, flip signs

    return hypervolume_1
# %%
# %%
if __name__ == "__main__":

    hvs = []
    optim_types = ['zinc','ctrl','random','chimera','hv']
    for optim_type in optim_types:
        hvs_type = [] 
        print(optim_type)

        if optim_type == 'zinc':
            zinc_hv = compute_pareto(optim_type)
            hvs_type.append(optim_type)
            hvs_type.append(zinc_hv*5)
            hvs.append((hvs_type))


        else:
            
            hv = compute_pareto(optim_type)
            hvs_type.append(optim_type)
            hvs_type.append(hv)

            for i in range(2,6):
                optim_string = optim_type + str(i)
                hv = compute_pareto(optim_string)
                hvs_type.append(hv)
            hvs.append((hvs_type))
    
    df_hvs = pd.DataFrame(hvs, columns =['optim_type','1','2','3','4','5'])
    df_hvs.to_csv("all_hvs.csv")




# %%


