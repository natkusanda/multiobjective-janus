# %%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP, MolMR
from SAS_calculator.sascorer import calculateScore

# %%
def normalize_and_sum(fitness_list):
    # %%
    norm_vals = 16.66 * fitness_list[:,0] + fitness_list[:,1]

    norm_idx = np.argsort(norm_vals)[::-1]

    return norm_idx, norm_vals
# %%
def scalarize_and_sort(scalarizer, fitness_list):
    """
    Parameters
    ----------
    scalarizer: Scalarizer() object
        Initialised scalarizer 

    fitness_list: np.array(float())

    Returns
    -------
    scalarizer_idx: list(int)
        Indices of inputted smiles_collector sorted by scalarizer
    scalarizer_vals: list(float)
        Scalarizer values, original indices, 0 (good) to 1 (bad)
    """    

    scalarizer_vals = scalarizer.scalarize(fitness_list)
    scalarizer_idx = np.argsort(scalarizer_vals)

    return scalarizer_idx, scalarizer_vals
# %%
def min_dist_utopia(all_fit):
    utopia = np.array([0.6,10])
    dist_1 = (all_fit[:,0] - utopia[0]) ** 2
    dist_2 = (all_fit[:,1] - utopia[1]) ** 2

    return min(dist_1 + dist_2)