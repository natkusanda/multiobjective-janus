"""
Filtering GDB-13
Authors: Robert Pollice, Akshat Nigam
Date: Sep. 2020
"""
# %%
import numpy as np
import pandas as pd
import rdkit as rd
from rdkit import Chem
import rdkit.Chem as rdc
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.Lipinski as rdcl
# import argparse as ap
import pathlib as pl

# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.drawOptions.addAtomIndices = True

cwd = pl.Path.cwd() # define current working directory

# %%
def substructure_violations(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """
    violation = False
    forbidden_fragments = [
        '[A;R]=[*;R2]', # Bredt's rule
        'a~[*;R2]~a', # Atom in aromatic ring connected to two rings
        '*1=**=*1', # Cyclobutadiene of all elements
        '*=*1*=***=*1', # Unstable atom double bond arrangment
        '[#6-]', # Negatively charged carbon
        '[#16-]', # Negatively charged sulphur
        '[#8-]', # Negatively charged oxygen
        '[#7-]', # Negatively charged nitrogen
        '[*+]', # Positively charged any atom, this makes the ones above redundant
        '[*-]', # Negatively charged any atom, this makes the ones above redundant
        '[#15;H]', # Phosphorous with hydrogen atom
        '[#15]', # Phosphorous atoms
        '[#7&X5]', # Pentavalent nitrogen
        '*=[#16;!R]', # Double bond to non-ring S
        '[#16&X3]', # Trivalent sulphur
        '[#16&X4]', # Tetravalent sulphur
        '[#16&X5]', # Pentavalent sulphur
        '[#16&X6]', # Hexavalent sulphur
        '[#5,#7,#8,#16]~[F,Cl,Br,I]', # Halogens connected to B, N, O or S
        '*=*=*', # Allene
        '*#*', # Triple bond 
        '[#8,#16]~[#8,#16]', # Bonds between O-O or S-S
        '[#7,#8,#16]~[#7,#8,#16]~[#7,#8,#16]', # N-O-S bonds including permutations
        '[#7,#8,#16]~[#7,#8,#16]~[#6]=,:[#7,#8,#16;!R]', # N-N-C=N bonds where last N is not in ring (N can be O or S as well)
        '*=N-[*;!R]', # *=N-* bond where last * is not in a ring 
        '*~[#7,#8,#16]-[#7,#8,#16;!R]', # *-N-N bonds where last N is not in a ring
        'c#ccccc', # benzene triple bond (originally separate "bad" filter)
        "[#5]", # Boron
        "[S;X2]", # non-aromatic sulfide
        "[c;$(c1nc(Br)cc([OH])c1),$(c1nc(Br)ccc([OH])1),$(c1nc(Br)c([OH])cc1),$(c1([OH])nc(Br)ccc1)]", # OH on pyridine ring
        "[N;H1]([OH])", # hydroxylamine group
        "[OX2H][#6X3]=[#6]", # enol
        "[C;X3]=[N;X2]", # imine group
        #"[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]", # substituted imine
        #"[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]", # substituted or unsubstituted imine
        "[NX3][CX3]=[CX3]", # enamine
        "C1C=C1", # cyclopropene
        "C1CC1", # cyclopropyl
        #"C=C-[N;H]" # cyclic enamine already captured in more general emanine
        "C1=CCC=C1" # cyclopentadiene
        ]
    for ni in range(len(forbidden_fragments)):
        if mol.HasSubstructMatch(rdc.MolFromSmarts(forbidden_fragments[ni])):
            violation = True
            # for indices in mol.GetSubstructMatches(rdc.MolFromSmarts(forbidden_fragments[ni])):
            #     display(Chem.Draw.MolToImage(mol, highlightAtoms = indices))
            # print(forbidden_fragments[ni])
            break
        else:
            continue
    return violation

# pyridine-related filters

def substructure_requirements(mol):
    """
    Check for substructure requirements
    Return True: contains required substructures
    Return False: doesn't contain required substructures
    """
    requirement = True
    required_fragments = [
        "c1[n;X2]c(Br)ccc1", # one pyridine reaction site
        "c1[c;H1][n;X2]c(Br)[c;H1]c1" # pyridine and no steric clash (pyridine + nucleophillic)
        #"c1[n;X2]c(Br)[c;H1]cc1", # pyridine and no steric clash with other pyridine
        #"c1[c;H1][n;X2]c(Br)cc1" # pyridine and no nucleophilic steric clash
        ]
    for ni in range(len(required_fragments)):
        if len(mol.GetSubstructMatches(rdc.MolFromSmarts(required_fragments[ni]))) != 1: # only 1 reaction site
            requirement = False
            #print(required_fragments[ni])
            break
        else:
            continue
    return requirement

def passes_filter(smi): 
    try:
        mol = Chem.MolFromSmiles(smi)

        # heavy atoms filter
        if mol.GetNumHeavyAtoms() > 50:
            return False

        # smiles length filter
        if len(smi) > 81 or len(smi) == 0:
            return False

        #if (substructure_requirements(mol) == False):
        #    return False
        if (substructure_violations(mol) == True):
            return False
        
        return True
    except:
        return False


