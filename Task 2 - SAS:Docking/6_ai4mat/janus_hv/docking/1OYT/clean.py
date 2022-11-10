import pickle
# import tqdm
import pandas as pd
# import rdkit.Chem as Chem
# 'CC(=O)Nc1ccc(S(=O)(=O)N2CCc3ccccc3[C@H]2CC(=O)NCc2cccc(C)c2)cc1'

data = pickle.load(open('COMBINED.pickle', 'rb'))
zinc = pd.read_csv('../../ZINC.csv')
zinc = zinc.set_index('smiles')

new_dict = {
    'smiles': [],
    'logP': [],
    'qed': [],
    'SAS': [],
    '1oyt': []
}
for smiles, vals in data.items():
    pre_calc = zinc.loc[smiles]
    new_dict['smiles'].append(smiles)
    new_dict['1oyt'].append(vals[0])
    new_dict['logP'].append(pre_calc['logP'])
    new_dict['qed'].append(pre_calc['qed'])
    new_dict['SAS'].append(pre_calc['SAS'])
    
    # import pdb; pdb.set_trace()

pd.DataFrame(new_dict).to_csv('ZINC_1oyt.csv', index=False)
