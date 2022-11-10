import os, sys
import multiprocessing
import random
from functools import partial
from typing import Callable, List

import pandas as pd
import numpy as np

from .crossover import crossover_smiles
from .mutate import mutate_smiles
from .network import obtain_model_pred, train_and_save_classifier
from .filter import passes_filter
from .utils import sanitize_smiles, get_fp_scores
from .janus_olympus import create_value_space, create_scalarizer, scalarize_and_sort, min_dist_utopia

# mpi4py
from mpi4py.futures import MPIPoolExecutor

class JANUS():
    ''' JANUS class for genetic algorithm applied on SELFIES
    string representation.
    '''

    def __init__(self, work_dir: str, fitness_function: Callable,
            custom_filter: Callable = None, alphabet: List = [],
            num_workers: int = None, 
            scalarizer: Callable = None, # NAT: MULTIOPT
            properties: List = [],
            objectives: List = [],
            kind: str = None,
            supplement: List = [], 
            **kwargs):
        print("initialising janus")
        # set all class variables
        self.work_dir = work_dir
        self.fitness_function = fitness_function
        self.custom_filter = custom_filter
        self.alphabet = alphabet
        
        # ---- NAT: START NEW MULTIOPT - New self attributes ----
        self.properties = properties
        self.objectives = objectives
        self.kind = kind
        self.supplement = supplement
        # ---- NAT: DONE MULTIOPT

        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers

        for key, val in kwargs.items():
            setattr(self, key, val)

        print("getting initial pop + fitness")
        # get initial population and fitnesses
        init_smiles, init_fitness = [], []
        orig_smiles = []
        with open(self.start_population, 'r') as f:
            for line in f:
                s_line = sanitize_smiles(line.strip())
                if s_line is not None:
                    init_smiles.append(s_line)
                    orig_smiles.append(line.strip())    # without sanitize

        # mpi4py to parallelize fitness function calls
        # with MPIPoolExecutor(self.num_workers) as executor:
        #     init_fitness = list(executor.map(
        #         self.fitness_function, 
        #         init_smiles
        #     )
        # )

        # hardcode the zinc results here (quick and dirty)
        zinc_df = pd.read_csv('ZINC_1oyt.csv')
        lut = zinc_df.set_index('smiles').to_dict('index')
        init_fitness = []
        for smi in orig_smiles:
            init_fitness.append([lut[smi]['SAS'], lut[smi]['1oyt']])
        init_fitness = np.array(init_fitness)
        
        '''
        idx = np.argsort(init_fitness)[::-1]
        init_smiles = np.array(init_smiles)[idx]
        init_fitness = np.array(init_fitness)[idx]
        self.population = init_smiles[:self.generation_size]
        self.fitness = init_fitness[:self.generation_size]
        print('fitness',self.fitness)

        if not os.path.isdir("./RESULTS/"):
            os.mkdir("./RESULTS/")

        # store in global
        self.smiles_collector = {}
        uniq_pop, idx, counts = np.unique(self.population, return_index = True, return_counts = True)
        for smi, count, i in zip(uniq_pop, counts, idx):
            self.smiles_collector.update({smi: [self.fitness[i], count]})
        '''

        ### ---- NAT: START MULTIOPT - Lines 71 - 87 ---- 
        self.scalarizer = create_scalarizer(self.properties, self.objectives, self.kind, self.supplement)

        # Initialising smiles collector for input into scalarizer
        # Later iterations: still use smiles_collector format? 
        # But not entire dictionary, new dictionary for each gen?
        
        # Don't need scalarizer values here (second return)
        idx, __ = scalarize_and_sort(self.scalarizer, np.array(init_fitness))
        # Same as original, but sorting based on scalarizer values
        init_smiles = np.array(init_smiles)[idx]
        init_fitness = np.array(init_fitness)[idx]
        self.population = init_smiles[:self.generation_size]
        self.fitness = init_fitness[:self.generation_size]

        if not os.path.isdir("./RESULTS/"):
            os.mkdir("./RESULTS/")

        self.smiles_collector = {}
        uniq_pop, idx, counts = np.unique(self.population, return_index = True, return_counts = True)
        for smi, count, i in zip(uniq_pop, counts, idx):
            self.smiles_collector.update({smi: [self.fitness[i], count]})
        ### ---- NAT: DONE MULTIOPT ----


    def mutate_smi_list(self, smi_list: List[str], space = 'local'):
        if space == 'local':
            num_random_samples = self.exploit_num_random_samples
            num_mutations = self.exploit_num_mutations
        elif space == 'explore':
            num_random_samples = self.explr_num_random_samples
            num_mutations = self.explr_num_mutations
        else:
            raise ValueError('Invalid space, choose "local" or "explore".')

        with MPIPoolExecutor(self.num_workers) as executor:
            mut_smi_list = list(executor.map(
                partial(
                    mutate_smiles,
                    alphabet = self.alphabet,
                    num_random_samples = num_random_samples,
                    num_mutations = num_mutations,
                    num_sample_frags=self.num_sample_frags_mutation
                ),
                smi_list
            )
        )
        mut_smi_list = self.flatten_list(mut_smi_list)
        return mut_smi_list

    def crossover_smi_list(self, smi_list: List[str]):
        with MPIPoolExecutor(self.num_workers) as executor:
            cross_smi = list(executor.map(
                partial(
                    crossover_smiles, 
                    crossover_num_random_samples = self.crossover_num_random_samples
                ), 
                smi_list
            )
        )
        cross_smi = self.flatten_list(cross_smi)
        return cross_smi

    def check_filters(self, smi_list: List[str]):
        if self.filter:
            smi_list = [smi for smi in smi_list if passes_filter(smi)]
        if self.custom_filter is not None:
            smi_list = [smi for smi in smi_list if self.custom_filter(smi)]
        return smi_list


    def run(self):

        for gen_ in range(self.generations):
        
            # bookkeeping
            if self.verbose_out:
                nn_tag = str(gen_) + '_NN'
                output_dir = os.path.join(self.work_dir, f'{gen_}_DATA')
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
            else:
                nn_tag = None       # do not save in subfolder named by generation index

            print(f"On generation {gen_}/{self.generations}")

            '''
            keep_smiles, replace_smiles = self.get_good_bad_smiles(
                self.fitness, 
                self.population, 
                self.generation_size
            )
            '''

            ### ---- NAT: START MULTIOPT - Lines 177 - 183 ----
            idx_temp, scalarizer_vals = scalarize_and_sort(self.scalarizer, np.array(self.fitness))
            print("Step 1 first sort: ", np.array(self.population)[idx_temp], np.array(self.fitness)[idx_temp])

            keep_smiles, replace_smiles = self.get_good_bad_smiles(
                -1 * scalarizer_vals, # 0-1 is best-worst
                self.population,
                self.generation_size
            )
            print("Smiles kept: ", len(keep_smiles))
            print("Smiles replaced: ", len(replace_smiles))
            
            ### ---- NAT: DONE MULTIOPT ----

            #print('population',self.population)

            ### EXPLORATION
            # Mutations:
            # Mutate and crossover (with keep_smiles) molecules that are meant to be replaced
            mut_smi_explr = self.mutate_smi_list(replace_smiles[0: len(replace_smiles)//2], space='explore')

            mut_smi_explr = self.check_filters(mut_smi_explr)

            print('mutate')
            # Crossovers:
            smiles_join = []

            for item in replace_smiles[len(replace_smiles) // 2 :]:
                smiles_join.append(item + "xxx" + random.choice(keep_smiles))
            cross_smi_explr = self.crossover_smi_list(smiles_join)

            cross_smi_explr = self.check_filters(cross_smi_explr)
            print('crossover')

            # Combine and get unique smiles
            all_smiles = list(set(mut_smi_explr + cross_smi_explr))
            all_smiles_unique = [x for x in all_smiles if x not in self.smiles_collector]

            # STEP 2: CONDUCT FITNESS CALCULATION FOR THE EXPLORATION MOLECULES:
            # Replace the molecules with ones in exploration mutation/crossover
            if gen_ == 0:
                #print(len(all_smiles_unique),self.generation_size,len(keep_smiles))
                replaced_pop = random.sample(
                    all_smiles_unique, self.generation_size - len(keep_smiles)
                )
            else:
                if self.use_NN_classifier == True:
                    # The sampling needs to be done by the neural network!
                    print("    Training classifier neural net...")
                    train_smiles, pro_val, fit_val = [], [], []
                    '''
                    for item in self.smiles_collector: 
                        train_smiles.append(item)
                        pro_val.append(self.smiles_collector[item][0])
                    train_and_save_classifier(train_smiles, pro_val, generation_index=nn_tag)
                    '''

                    ### ---- NAT: START MULTIOPT - Lines 240 - 245 ----
                    for item in self.smiles_collector: 
                        train_smiles.append(item)
                        fit_val.append(self.smiles_collector[item][0])

                    __, scalarizer_vals = scalarize_and_sort(self.scalarizer, np.array(fit_val))
                    for i in range(len(train_smiles)): 
                        #pro_val.append(-1 * scalarizer_vals[i]) # 0 to 1 : good to bad
                        pro_val.append(scalarizer_vals[i])
                    train_and_save_classifier(train_smiles, pro_val, generation_index=nn_tag)
                    ### ---- NAT: END MULTIOPT ----

                    # Obtain predictions on unseen molecules:
                    ### ---- NAT: Multiopt predicting scalarizer values, sorting based on scalarizers
                    print("    Obtaining Predictions")
                    new_predictions = obtain_model_pred(
                        all_smiles_unique, "classifier", generation_index=nn_tag
                    )
                    ### ---- NAT: Multiopt - if predicted values are 0 to 1 good to bad, no need to reverse order
                    #NN_pred_sort = np.argsort(new_predictions)[::-1]
                    NN_pred_sort = np.argsort(new_predictions)
                    replaced_pop = [
                        all_smiles_unique[NN_pred_sort[i]]
                        for i in range(self.generation_size - len(keep_smiles))
                    ]
                else:
                    replaced_pop = random.sample(
                        all_smiles_unique, self.generation_size - len(keep_smiles)
                    )

            # Calculate actual fitness for the exploration population
            self.population = keep_smiles + replaced_pop
            self.fitness = []

            '''
            for smi in self.population:
                if smi in self.smiles_collector:
                    # if already calculated, use the value from smiles collector
                    self.fitness.append(self.smiles_collector[smi][0])
                    self.smiles_collector[smi][1] += 1
                else:
                    # make a calculation
                    f = self.fitness_function(smi)
                    self.fitness.append(f)
                    self.smiles_collector[smi] = [f, 1]
            '''

            # ---- Nat: START PARALLELISING 276 - 287 ----
            new_pop_smiles = []
            for smi in self.population:
                if smi in self.smiles_collector:
                    # if already calculated, use the value from smiles collector
                    self.fitness.append(self.smiles_collector[smi][0])
                    self.smiles_collector[smi][1] += 1
                else:
                    new_pop_smiles.append(smi)

            with MPIPoolExecutor(self.num_workers) as executor:
                new_pop_fitness = list(executor.map(
                    self.fitness_function, 
                    new_pop_smiles
                )
            )
            # FIX FORMATTING? np.array vs tuple
            new_pop_fitness = np.array(new_pop_fitness)

            for new_pop_i in range(len(new_pop_smiles)):
                f = new_pop_fitness[new_pop_i]
                self.fitness.append(f)
                self.smiles_collector[new_pop_smiles[new_pop_i]] = [f, 1]
            # ---- Nat: DONE PARALLELISING ----

            '''
            idx_sort = np.argsort(self.fitness)[::-1]
            '''

            ### ---- NAT: START MULTIOPT - Line 312 ----
            # At this point, smiles_collector and self.population DO NOT have the same smiles
            # Create a dictionary with just the smiles in self.population
            
            idx_sort, scalarizer_vals = scalarize_and_sort(self.scalarizer, np.array(self.fitness))
            ### ---- NAT: END MULTIOPT ----

            print(f"    (Explr) Top Fitness: {self.fitness[idx_sort[0]]}")
            print(f"    (Explr) Top Smile: {self.population[idx_sort[0]]}")

            # ---- NAT - Sorting here is now based on scalarizer values
            # ---- NAT - add sorted scalarizer for writing later
            # Fitnesses written are now tuples
            fitness_sort = np.array(self.fitness)[idx_sort]
            scalarizer_sort = np.array(scalarizer_vals)[idx_sort]

            if self.verbose_out:
                with open("./RESULTS/" + str(gen_) + "_DATA/fitness_explore.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])
                # ---- NAT: MULTIOPT PRINTOUT
                with open("./RESULTS/" + str(gen_) + "_DATA/scalarizer_explore.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in scalarizer_sort])
                    f.writelines(["\n"])
            else:
                with open("./RESULTS/fitness_explore.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])
                # ---- NAT: MULTIOPT PRINTOUT
                with open("./RESULTS/scalarizer_explore.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in scalarizer_sort])
                    f.writelines(["\n"])

            # this population is sort by modified fitness, if active
            # ---- NAT - Sorting here is now based on scalarizer values
            population_sort = np.array(self.population)[idx_sort]
            if self.verbose_out:
                with open("./RESULTS/" + str(gen_) + "_DATA/population_explore.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])
            else:
                with open("./RESULTS/population_explore.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])
            
            print("Step 2 explore sort:", population_sort,fitness_sort)
            # STEP 3: CONDUCT LOCAL SEARCH ON EXPLOITATION SET:
            # Conduct local search on top-n molecules from population, mutate and do similarity search
            smiles_local_search = population_sort[0 : self.top_mols]
            # maybe need to increase top_mols?

            mut_smi_loc = self.mutate_smi_list(smiles_local_search, 'local')
            mut_smi_loc = self.check_filters(mut_smi_loc)

            # filter out molecules already found
            mut_smi_loc = [x for x in mut_smi_loc if x not in self.smiles_collector]

            # sort by similarity, only keep ones similar to best
            fp_scores = get_fp_scores(mut_smi_loc, population_sort[0])
            fp_sort_idx = np.argsort(fp_scores)[::-1][: self.generation_size]

            # highest fp_score idxs
            self.population_loc = np.array(mut_smi_loc)[fp_sort_idx]  # list of smiles with highest fp scores

            # STEP 4: CALCULATE THE FITNESS FOR THE LOCAL SEARCH:
            # Exploitation data generated from similarity search is measured with fitness function
            # and modified with discriminator score if stagnation patience reached at this generation
            self.fitness_loc = []

            """
            for smi in self.population_loc:
                # if smi in self.smiles_collector:    # could get rid of this condition (checked in line 220)
                #     self.fitness_loc.append(self.smiles_collector[smi][0])
                #     self.smiles_collector[smi][1] += 1
                # else:
                f = self.fitness_function(smi)
                self.fitness_loc.append(f)
                self.smiles_collector[smi] = [f, 1]
            """

            # ---- Nat: START PARALLELISING 386 - 395 ----
            new_loc_smiles = self.population_loc
 
            with MPIPoolExecutor(self.num_workers) as executor:
                new_loc_fitness = list(executor.map(
                    self.fitness_function, 
                    new_loc_smiles
                )
            )
            # FIX FORMATTING? np.array vs tuple
            new_loc_fitness = np.array(new_loc_fitness)

            for new_loc_i in range(len(new_loc_smiles)):
                f = new_loc_fitness[new_loc_i] 
                self.fitness_loc.append(f)
                self.smiles_collector[new_loc_smiles[new_loc_i]] = [f, 1]
                
            # ---- Nat: END PARALLELISING ----
            

            #print("length of fitness loc:", len(self.fitness_loc))
            # List of original local fitness scores
            '''
            idx_sort = np.argsort(self.fitness_loc)[::-1]         # index of highest to lowest fitness scores
            '''
            ### ---- NAT: START MULTIOPT Line 416 ----
           
            idx_sort, scalarizer_vals = scalarize_and_sort(self.scalarizer, np.array(self.fitness_loc))
            ### ---- NAT: END MULTIOPT ----

            print(f"    (Local) Top Fitness: {self.fitness_loc[idx_sort[0]]}")
            print(f"    (Local) Top Smile: {self.population_loc[idx_sort[0]]}")

            # ---- NAT - Sorting here is now based on scalarizer values
            # ---- NAT - add sorted scalarizer for writing later
            # Fitnesses written are now tuples
            fitness_sort = np.array(self.fitness_loc)[idx_sort]
            scalarizer_sort = np.array(scalarizer_vals)[idx_sort]

            if self.verbose_out:
                with open(
                    "./RESULTS/" + str(gen_) + "_DATA/fitness_local_search.txt", "w"
                ) as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])
                # ---- NAT: MULTIOPT PRINTOUT
                with open("./RESULTS/" + str(gen_) + "_DATA/scalarizer_local_search.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in scalarizer_sort])
                    f.writelines(["\n"])
            else:
                with open("./RESULTS/fitness_local_search.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])
                with open("./RESULTS/scalarizer_local_search.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in scalarizer_sort])
                    f.writelines(["\n"])

            population_sort = np.array(self.population_loc)[idx_sort]
            if self.verbose_out:
                with open(
                    "./RESULTS/" + str(gen_) + "_DATA/population_local_search.txt", "w"
                ) as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])
            else:
                with open("./RESULTS/population_local_search.txt", "w") as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])
            
            print("Step 4 local sort:", population_sort, fitness_sort)
            # STEP 5: EXCHANGE THE POPULATIONS:
            # Introduce changes to 'fitness' & 'population'
            best_smi_local = population_sort[0:self.num_exchanges]
            best_fitness_local = fitness_sort[0:self.num_exchanges]

            '''
            # But will print the best fitness values in file
            idx_sort = np.argsort(self.fitness)[::-1]         # sorted indices for the entire population
            worst_indices = idx_sort[-self.num_exchanges:]    # replace worst ones with the best ones
            for i, idx in enumerate(worst_indices):
                try:
                    self.population[idx] = best_smi_local[i]
                    self.fitness[idx] = best_fitness_local[i]
                except:
                    continue

            # Save best of generation!:
            fit_all_best = np.argmax(self.fitness)
            '''
            ### ---- NAT: START MULTIOPT - Lines 469 - 482 ----
            
            idx_sort, __ = scalarize_and_sort(self.scalarizer, np.array(self.fitness))
            print("Step 5 cur sort:", np.array(self.population)[idx_sort], np.array(self.fitness)[idx_sort])
            worst_indices = idx_sort[-self.num_exchanges:]    # replace worst ones with the best ones
            for i, idx in enumerate(worst_indices):
                try:
                    self.population[idx] = best_smi_local[i]
                    self.fitness[idx] = best_fitness_local[i]
                except:
                    continue
            # Save best of generation!: 
            # NEED TO MAKE NEW COLLECTOR WITH ADDED BEST LOCAL SMILES

            idx_sort, __ = scalarize_and_sort(self.scalarizer, np.array(self.fitness))
            print("Step 5 fin sort:", np.array(self.population)[idx_sort], np.array(self.fitness)[idx_sort])
            fit_all_best = idx_sort[0] # Sorting based on scalarized values
            fitness_sort = np.array(self.fitness)[idx_sort]
            
            # verify population against gen_best
            with open(
                    "./RESULTS/" + str(gen_) + "_DATA/final_gen_fitness.txt", "w"
                ) as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])

            ### ---- NAT: END MULTIOPT ----

            ### ---- NAT: COMPUTE DISTANCE TO UTOPIA POINT
            min_dist = min_dist_utopia(fitness_sort)
            with open("./RESULTS" + "/generation_min_dist.txt", "a+") as f:
                f.writelines(f"Gen:{gen_}, {min_dist} \n")


            # write best molecule with best fitness
            with open("./RESULTS" + "/generation_all_best.txt", "a+") as f:
                f.writelines(f"Gen:{gen_}, {self.population[fit_all_best]}, {self.fitness[fit_all_best]} \n")

            if gen_ == self.generations - 1:
                all_fit = []
                all_smiles = []
                for item in self.smiles_collector: 
                    all_smiles.append(item)
                    all_fit.append(self.smiles_collector[item][0])
                idx_sort, scalarizer_vals = scalarize_and_sort(self.scalarizer, np.array(all_fit))
                fitness_sort = np.array(all_fit)[idx_sort]
                scalarizer_sort = np.array(scalarizer_vals)[idx_sort]
                smiles_sort = np.array(all_smiles)[idx_sort]

                with open("./RESULTS" + "/smiles_collector.txt", "a+") as f:
                    f.writelines(["{},{},{} \n".format(i,x,y) for i,x,y in zip(range(len(smiles_sort)),smiles_sort,fitness_sort)])

                csv_vals = []
                for j in range(len(smiles_sort)):
                    csv_vals.append((j,smiles_sort[j],fitness_sort[j][0],fitness_sort[j][1]))
                df = pd.DataFrame(csv_vals, columns =['i','smi','qed','logp'])
                df.to_csv("smiles_collector.csv")



        return

    @staticmethod
    def get_good_bad_smiles(fitness, population, generation_size):
        """
        Given fitness values of all SMILES in population, and the generation size, 
        this function smplits  the population into two lists: keep_smiles & replace_smiles. 
        
        Parameters
        ----------
        fitness : (list of floats)
            List of floats representing properties for molecules in population.
        population : (list of SMILES)
            List of all SMILES in each generation.
        generation_size : (int)
            Number of molecules in each generation.

        Returns
        -------
        keep_smiles : (list of SMILES)
            A list of SMILES that will be untouched for the next generation. .
        replace_smiles : (list of SMILES)
            A list of SMILES that will be mutated/crossed-oved for forming the subsequent generation.

        """

        fitness = np.array(fitness)
        idx_sort = fitness.argsort()[::-1]  # Best -> Worst
        keep_ratio = 0.2
        keep_idx = int(len(list(idx_sort)) * keep_ratio)
        try:

            F_50_val = fitness[idx_sort[keep_idx]]
            F_25_val = np.array(fitness) - F_50_val
            F_25_val = np.array([x for x in F_25_val if x < 0]) + F_50_val
            F_25_sort = F_25_val.argsort()[::-1]
            F_25_val = F_25_val[F_25_sort[0]]

            prob_ = 1 / (3 ** ((F_50_val - fitness) / (F_50_val - F_25_val)) + 1)

            prob_ = prob_ / sum(prob_)
            # replace=False added to avoid duplicates
            to_keep = np.random.choice(generation_size, keep_idx, p=prob_, replace=False)
            to_replace = [i for i in range(generation_size) if i not in to_keep][
                0 : generation_size - len(to_keep)
            ]

            keep_smiles = [population[i] for i in to_keep]
            replace_smiles = [population[i] for i in to_replace]

            best_smi = population[idx_sort[0]]
            if best_smi not in keep_smiles:
                keep_smiles.append(best_smi)
                if best_smi in replace_smiles:
                    replace_smiles.remove(best_smi)

            if keep_smiles == [] or replace_smiles == []:
                raise Exception("Badly sampled population!")
        except:
            keep_smiles = [population[i] for i in idx_sort[:keep_idx]]
            replace_smiles = [population[i] for i in idx_sort[keep_idx:]]

        return keep_smiles, replace_smiles

    def log(self):
        pass

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]


