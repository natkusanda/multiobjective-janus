import os, sys
import csv
import time
import tempfile, inspect

import selfies as sf
import selfies
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import multiprocessing
from functools import partial
import subprocess

def get_docking_score():
    with open('./DOCKING_TEST_log.txt', 'r') as f:
        lines = f.readlines()

    lines = lines[25].split(' ')
    lines = [x for x in lines if len(x) > 0]
    lines = float(lines[1])

    return lines

def fitness_function(smi: str, target: str = '1OYT'):
    cwd = os.getcwd()
    tmp_dir = tempfile.TemporaryDirectory(dir='/tmp')
    os.chdir(tmp_dir.name)

    smina_path = os.path.join(cwd, 'docking')
    target_path = os.path.join(smina_path, target)

    # name = re.sub(r'[^\w]', '', smi)
    # name = 'mol'
    t0 = time.time()

    mol = Chem.MolFromSmiles(smi)
    try:
        sascore = sascorer.calculateScore(mol)
    except:
        sascore = 10.0

    # generate ligand files directly from smiles using openbabel
    # print('Default to openbabel embedding...')
    with open('./test.smi', 'w') as f:
        f.writelines('{}'.format(smi))

    # openbabel embedding
    _ = subprocess.run(f'obabel test.smi --gen3d -O test.sdf', shell=True, capture_output=True)
    os.system('rm test.smi')

    try:
        # run docking procedure
        output = subprocess.run(f"{smina_path}/smina.static -r {target_path}/receptor.pdb -l test.sdf --autobox_ligand \
            {target_path}/ligand.pdb --autobox_add 3 --exhaustiveness 10 -o DOCKING_TEST.pdb --log DOCKING_TEST_log.txt",
            shell=True, capture_output=True)

        if output.returncode != 0:
            # print('Job failed.')
            score = 999.0
        else:
            # extract result from output
            score = get_docking_score()
    except:
        score = 999.0
    
    # write to file
    # with open(os.path.join(target_path, 'cache.csv'), 'a') as f:
    #     f.write(f'{smi},{score},{time.time() - t0}\n')

    with open(os.path.join(cwd, 'OUT_ALL.csv'), 'a') as f:
        f.write(f'{smi},{sascore},{score},{time.time() - t0}\n')

    os.system('rm test.sdf')
    os.chdir(cwd)
    tmp_dir.cleanup()
        
    return (sascore, score)

