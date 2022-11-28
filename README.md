# Multiobjective JANUS: Genetic Algothirm Augmented with Scalarizers for Multiobjective Optimization
This repository containis the relevant code and experiments for the paper [Assessing multi-objective optimization of molecules with genetic algorithms against relevant baselines](https://openreview.net/forum?id=sWRZxIcR8qK). 

The algorithm can be run by initialising a file similar ```Click.py```, which is reproduced in part below. Most importantly, this involves defining a fitness function that outputs a tupple of properties, as well as defining  ```properties```, ```objectives```, ```kind```to  denote maximization or minimization, as well as the type of scalarizer needed. Options available are Chimera, Hypervolume and WeightedSum.

A ```supplement``` variable is required for Chimera – where it denotes thresholds the optimization should aim to meet for each objective – and WeightedSum, wherein it denotes relative weights of objectives.

```python
def fitness_function(smi):
    """
    Parameters
    ----------
    smi: SMILES string
        For one molecule
    Returns
    -------
    fitnesses: tuple(float)
        Tuple of all fitness values computed
    """    
    
    mol = Chem.MolFromSmiles(smi)

    sas_val = calculateScore(mol)
    logp_val = MolLogP(mol)
    qed_val = qed(mol)

    fitnesses = (qed_val, logp_val, sas_val)

    return fitnesses
    
def main():
    print('name=main')
    params = generate_params()

    properties = ['qed','logp', 'sas']
    objectives = ['max', 'max', 'min']
    kind = 'Chimera'
    supplement = [0.6,8,0]

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
 ```
    

