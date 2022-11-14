
# %%
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import numpy as np

import olympus
from olympus.utils.misc import get_hypervolume, get_pareto, get_pareto_set, get_r2
import plotly.express as px
import plotly.graph_objects as go
# %%
def compute_pareto(optim_type, w_ref, mode):
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
    df.to_csv(optim_type + "_pareto_tagged_logp_qed_sas.csv")
    print("logp, qed, sas Done!")

    new_ref = np.array([-w_ref[0], -w_ref[1], w_ref[2]])
    if mode == 'r2':
        volume = get_r2(pareto_front, new_ref)
    else:
        volume = get_hypervolume(pareto_front, new_ref)
    # get_r2, flip signs

    return volume
# %%
def compute_w_ref(optim_type):
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
    
    return [max(logp), max(qed), min(sas)]

# %%
if __name__ == "__main__":

    hvs = []
    optim_types = ['zinc','ctrl','random','chimera','newchim','hv']

    mode = 'r2'
    #mode = 'hv'
    if mode == 'r2':
        abs_ref = [0,0,1000]
        for optim_type in optim_types:
            hvs_type = [] 
            print(optim_type)

            if optim_type == 'zinc':
                find_ref = compute_w_ref(optim_type)
                abs_ref = [max(abs_ref[0],find_ref[0]),max(abs_ref[1],find_ref[1]),min(abs_ref[2],find_ref[2])]

            else:
                find_ref = compute_w_ref(optim_type)
                abs_ref = [max(abs_ref[0],find_ref[0]),max(abs_ref[1],find_ref[1]),min(abs_ref[2],find_ref[2])]

                for i in range(2,11):
                    optim_string = optim_type + str(i)
                    find_ref = compute_w_ref(optim_string)
                    abs_ref = [max(abs_ref[0],find_ref[0]),max(abs_ref[1],find_ref[1]),min(abs_ref[2],find_ref[2])]
    else:
        logp_worst = -6.8762
        qed_worst = 0.111811475018
        sas_worst = 7.289282840617412
        abs_ref = [logp_worst, qed_worst, sas_worst]
    print(abs_ref)

    for optim_type in optim_types:
        hvs_type = [] 
        print(optim_type)

        if optim_type == 'zinc':
            zinc_hv = compute_pareto(optim_type, abs_ref, mode)
            hvs_type.append(optim_type)
            hvs_type.append(zinc_hv*10)
            hvs.append((hvs_type))

        else:
            hv = compute_pareto(optim_type, abs_ref, mode)
            hvs_type.append(optim_type)
            hvs_type.append(hv)

            for i in range(2,11):
                optim_string = optim_type + str(i)
                hv = compute_pareto(optim_string, abs_ref, mode)
                hvs_type.append(hv)
            hvs.append((hvs_type))
    
    df_hvs = pd.DataFrame(hvs, columns =['optim_type','1','2','3','4','5','6','7','8','9','10'])
    df_hvs.to_csv("all_" + mode + ".csv")




# %%


