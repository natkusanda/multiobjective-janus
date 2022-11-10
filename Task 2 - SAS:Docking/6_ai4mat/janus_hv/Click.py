import multiprocessing
from janus import JANUS
# %%
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
RDLogger.DisableLog("rdApp.*")

from dock import fitness_function
import selfies
import sys
import pandas as pd
import numpy as np

# SAScore, LogP, cycles, QED import
# from rdkit.Chem.Crippen import MolLogP, MolMR
# from SAS_calculator.sascorer import calculateScore
# from rdkit.Chem.Descriptors import qed

# %%
def generate_params():
    """
    Parameters for initiating JANUS. The parameters here are picked based on prior 
    experience by the authors of the paper. 
    """

    params_ = {}

    # Record data from every generation in individual directories
    params_["verbose_out"] = True

    # Number of iterations that JANUS runs for:
    params_["generations"] = 100  # 200

    # The number of molecules for which fitness calculations are done, within each generation
    params_["generation_size"] = 100  # 5000

    # Location of file containing SMILES that will be user for the initial population.
    # NOTE: number of smiles must be greater than generation size.
    # params_["start_population"] = "./DATA/C#C_STONED_fixed_220505.txt"
    params_["start_population"] = "./DATA/zinc_250k_sliced.txt"

    # Number of molecules that are exchanged between the exploration and exploitation
    # componenets of JANUS.
    params_["num_exchanges"] = 2

    # An option to generate fragments and use then when performing mutations.
    # Fragments are generated using the SMILES provided for the starting population.
    # The list of generated fragments is stored in './DATA/fragments_selfies.txt'
    params_["use_fragments"] = True  # Set to true

    # An option to use a classifier for sampling. If set to true, the trailed model
    # is saved at the end of every generation in './RESULTS/'.
    params_["use_NN_classifier"] = False  # Set this to true!

    # Number of top molecules to conduct local search
    params_["top_mols"] = 10

    # Number of randomly sampled SELFIE strings from alphabet
    params_["num_sample_frags_mutation"] = 100

    # Number of samples from random mutations in exploration population
    params_["explr_num_random_samples"] = 100

    # Number of random mutations in exploration population
    params_["explr_num_mutations"] = 100

    # Number of samples from random mutations in exploitation population
    params_["exploit_num_random_samples"] = 100

    # Number of random mutations in exploitation population
    params_["exploit_num_mutations"] = 100

    # Number of random crossovers
    params_["crossover_num_random_samples"] = 5  # 1

    # Use discriminator to modify fitness
    params_["use_NN_discriminator"] = False

    # Optional filter to ensure mutations do not create unwanted molecular structures
    params_["filter"] = True
    print('params')
    return params_

# def fitness_function(smi):
#     """
#     Parameters
#     ----------
#     smi: SMILES string
#         For one molecule

#     Returns
#     -------
#     fitnesses: tuple(float)
#         Tuple of all fitness values computed
#     """    
    
#     mol = Chem.MolFromSmiles(smi)

#     #sas_val = calculateScore(mol)
#     logp_val = MolLogP(mol)
#     #cycle_val = num_long_cycles(mol)
#     qed_val = qed(mol)

#     #fitnesses = (logp_val, sas_val, qed_val)
#     fitnesses = (qed_val, logp_val)

#     return fitnesses
'''
def num_long_cycles(mol):
    """Calculate the number of long cycles.
    Args:
        mol: Molecule. A molecule.
    Returns:
        negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if not cycle_list:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length
'''
def main():
    print('name=main')
    params = generate_params()

    #properties = ['logp', 'sascore', 'qed']
    #objectives = ['max', 'min', 'max']
    properties = ['SAS', '1oyt']
    objectives = ['min', 'min']
    kind = 'Hypervolume'
    #supplement = [1000,5,1000]
    supplement = [1000,1000]

    agent = JANUS(
        work_dir='RESULTS', 
        num_workers = 192,
        fitness_function = fitness_function,
        properties = properties,
        objectives = objectives,
        kind = kind,
        supplement = supplement, 
        **params
    )

    agent.run()
    print('done!')

if __name__ == '__main__':
    main()



# %%
