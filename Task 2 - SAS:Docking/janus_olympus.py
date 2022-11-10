# %%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import olympus
from olympus.datasets import Dataset
from olympus.campaigns import ParameterSpace
from olympus.objects import ParameterContinuous
from olympus.scalarizers import Scalarizer

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP, MolMR

# from rdkit.Chem import RDConfig
# import os, sys
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# from sascorer import calculateScore
# from SAS_calculator.sascorer import calculateScore
import random


# %%
def create_value_space(properties):
    """
    Parameters
    ----------
    properties : list(str)
        List of property objective names

    Returns
    -------
    value_space : ParameterSpace() object
        Value space of objectives for input in create_scalarizer
    """

    value_space = ParameterSpace()

    for i in range(len(properties)):
        prop = ParameterContinuous(name=properties[i])
        value_space.add(prop)

    return value_space
# %%
def create_scalarizer(properties, objectives, kind, supplement):
    """
    Parameters
    ----------
    properties : list(str)
        List of property objective names
        Used to generated empty value_space object
    
    objectives: list(str)
        List of optimisation objectives ('max' / 'min')
    
    kind: str
        Chimera, WeightedSum, Parego, Hypervolume
    
    supplement: list / int
        For Chimera, these are the tolerances / thresholds for each obejctive
        For WeightedSum, these are the weights
        For Parego, this is the rho value

    Dependencies
    ------------
    create_value_space()

    Returns
    -------
    scalarizer: Scalarizer() object
        Scalarizer object for scalarizing of dataset
    """

    value_space = create_value_space(properties)

    if kind == 'Chimera':
        scalarizer = Scalarizer(
            kind = 'Chimera',
            value_space = value_space,
            goals = objectives,
            tolerances = supplement,
            absolutes = [True] * len(properties)
        )
    elif kind == 'WeightedSum':
        scalarizer = Scalarizer(
            kind = 'WeightedSum',
            value_space = value_space,
            goals = objectives,
            weights = supplement
        )
    elif kind == 'Parego':
        scalarizer = Scalarizer(
            kind = 'Parego',
            value_space = value_space,
            goals = objectives,
            rho = supplement[0]
        )
    elif kind == 'Hypervolume':
        scalarizer = Scalarizer(
            kind = 'Hypervolume',
            value_space = value_space,
            goals = objectives
        )
    return scalarizer
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
    print(scalarizer_vals)
    scalarizer_idx = np.argsort(scalarizer_vals)

    return scalarizer_idx, scalarizer_vals



