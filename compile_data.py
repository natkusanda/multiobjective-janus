
# %%
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdChemReactions
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import numpy as np
from pathlib import Path

# LogP, QED import
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import qed

import olympus
from olympus.utils.misc import get_hypervolume, get_pareto, get_pareto_set
from janus_olympus import create_value_space, create_scalarizer, scalarize_and_sort
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
# %%
# %% Reading Pareto-tagged zinc dataset
df = pd.read_csv("pareto_tagged_logp_qed.csv")
smiles = list(df['smiles'])
logp = list(df['logP'])
qed = list(df['qed'])
pareto = list(df['pareto'])
idx = range(len(smiles))
# %% Getting hypervolume of Pareto front for Zinc dataset
logp_front = [logp[i] for i in range(len(idx)) if pareto[i] == 1]
qed_front = [qed[i] for i in range(len(idx)) if pareto[i] == 1]
pareto_front = np.column_stack((logp_front,qed_front))
pareto_front[:,0] = - pareto_front[:,0]
pareto_front[:,1] = - pareto_front[:,1]
logp_worst = -6.8762
qed_worst = 0.025882
w_ref = np.array([-logp_worst,-qed_worst])
hypervolume = get_hypervolume(pareto_front, w_ref)
print(hypervolume)
# %%
# Task 1: getting hypervolumes, retain 10% percentile worst molecules
properties = ['logp', 'qed']
objectives = ['max', 'max']
kind = 'Hypervolume'
supplement = [0,0]
data = np.column_stack((logp,qed))
hv_scalarizer = create_scalarizer(properties, objectives, kind, supplement)
idx, scal_vals = scalarize_and_sort(hv_scalarizer, data)
# %%
hv_scal_avg = np.mean(scal_vals)
hv_scal_90 = np.percentile(scal_vals,90)
hv_keep_data = []
hv_keep_smiles = []
for i in range(len(scal_vals)):
    if scal_vals[i] > hv_scal_90:
        hv_keep_data.append(data[i])
        hv_keep_smiles.append(smiles[i])
hv_keep_data = np.array(hv_keep_data)
# %%
with open(pwd + "zinc_250k_sliced.txt", "w") as f:
    f.writelines(["{}".format(x) for x in hv_keep_smiles])
# %%
# Task 2: Getting minimum distances to utopian point
def gen_min_all(optim_type):
    pwd = "/RESULTS_" + optim_type 
    min_distances = []
    min_dist_abs = 1000

    for i in tqdm(range(100)):
        my_file = open(pwd +  "/" + str(i) + "_DATA/" + "final_gen_fitness.txt", 'r')
        content = my_file.read()
        content_list = content.split()
        my_file.close()

        logp = []
        qed = []

       # Local search
        my_file = open(pwd +  "/" + str(i) + "_DATA/" + "fitness_local_search.txt", 'r')
        content = my_file.read()
        content_list = content.split()
        my_file.close()

        d = 0
        for j in range(len(content_list)):
            if (j+3-d) % 3 == 0:
                if len(content_list[j]) > 1:
                    if content_list[j][0] == '[':
                        qed.append(float(content_list[j][1:]))
                    else:
                        qed.append(float(content_list[j]))
                else:
                    d = d + 1
                    continue
            if (j+2-d) % 3 == 0:
                if content_list[j][-1] == ']':
                    logp.append(float(content_list[j][:-1]))
                    d = d - 1
                    continue
                else:
                    logp.append(float(content_list[j]))
        
        # Explore search
        my_file = open(pwd +  "/" + str(i) + "_DATA/" + "fitness_explore.txt", 'r')
        content = my_file.read()
        content_list = content.split()
        my_file.close()

        d = 0
        for j in range(len(content_list)):
            if (j+3-d) % 3 == 0:
                if len(content_list[j]) > 1:
                    if content_list[j][0] == '[':
                        qed.append(float(content_list[j][1:]))
                    else:
                        qed.append(float(content_list[j]))
                else:
                    d = d + 1
                    continue
            if (j+2-d) % 3 == 0:
                if content_list[j][-1] == ']':
                    logp.append(float(content_list[j][:-1]))
                    d = d - 1
                    continue
                else:
                    logp.append(float(content_list[j]))

        data = np.column_stack((qed,logp))
        utopia = np.array([0.6,10])
        dist_1 = (utopia[0] - data[:,0]) / utopia[0]
        dist_2 = (utopia[1] - data[:,1]) / utopia[1]

        # For maximization
        dist_1 = np.array([dist_1[i] if dist_1[i]>0 else 0 for i in range(len(dist_1))])
        dist_2 = np.array([dist_2[i] if dist_2[i]>0 else 0 for i in range(len(dist_2))])

        all_dist = np.sqrt(dist_1 ** 2 + dist_2 ** 2)
        min_dist = min(all_dist)
        if min_dist < min_dist_abs:
            min_dist_abs = min_dist
        min_distances.append(min_dist_abs)

    return min_distances
# %%
chim_min = gen_min_all('chimera')
ctrl_min = gen_min_all('ctrl')
hv_min = gen_min_all('hv')
rand_min = gen_min_all('random')
# %%
def stack_mean_std(big_mat):
    mindist_avg = np.mean(big_mat,axis = 0)
    mindist_std = np.std(big_mat,axis = 0)
    mindist = np.column_stack((mindist_avg,mindist_std))

    return mindist
# %%
chimera_min = np.empty((10,100))
chimera_min[0] = gen_min_all("chimera")
chimera_min[1] = gen_min_all("chimera2")
chimera_min[2] = gen_min_all("chimera3")
chimera_min[3] = gen_min_all("chimera4")
chimera_min[4] = gen_min_all("chimera5")
chimera_min[5] = gen_min_all("chimera6")
chimera_min[6] = gen_min_all("chimera7")
chimera_min[7] = gen_min_all("chimera8")
chimera_min[8] = gen_min_all("chimera9")
chimera_min[9] = gen_min_all("chimera10")
chim_mindist = stack_mean_std(chimera_min)
# %%
ctrl_min = np.empty((10,100))
ctrl_min[0] = gen_min_all("ctrl")
ctrl_min[1] = gen_min_all("ctrl2")
ctrl_min[2] = gen_min_all("ctrl3")
ctrl_min[3] = gen_min_all("ctrl4")
ctrl_min[4] = gen_min_all("ctrl5")
ctrl_min[5] = gen_min_all("ctrl6")
ctrl_min[6] = gen_min_all("ctrl7")
ctrl_min[7] = gen_min_all("ctrl8")
ctrl_min[8] = gen_min_all("ctrl9")
ctrl_min[9] = gen_min_all("ctrl10")
ctrl_mindist = stack_mean_std(ctrl_min)
# %%
hv_min = np.empty((10,100))
hv_min[0] = gen_min_all("hv")
hv_min[1] = gen_min_all("hv2")
hv_min[2] = gen_min_all("hv3")
hv_min[3] = gen_min_all("hv4")
hv_min[4] = gen_min_all("hv5")
hv_min[5] = gen_min_all("hv6")
hv_min[6] = gen_min_all("hv7")
hv_min[7] = gen_min_all("hv8")
hv_min[8] = gen_min_all("hv9")
hv_min[9] = gen_min_all("hv10")
hv_mindist = stack_mean_std(hv_min)
# %%
rand_min = np.empty((10,100))
rand_min[0] = gen_min_all("random")
rand_min[1] = gen_min_all("random2")
rand_min[2] = gen_min_all("random3")
rand_min[3] = gen_min_all("random4")
rand_min[4] = gen_min_all("random5")
rand_min[5] = gen_min_all("random6")
rand_min[6] = gen_min_all("random7")
rand_min[7] = gen_min_all("random8")
rand_min[8] = gen_min_all("random9")
rand_min[9] = gen_min_all("random10")
rand_mindist = stack_mean_std(rand_min)
# %%
def plot_shade(y1,y2,y3,y4,ext):
    t = np.arange(1, 101, 1)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xlim([1,100])
    ax.set_ylim([0,0.6])

    ax.plot(t, y1[:,0], color = 'C0')
    ax.plot(t, y2[:,0], color = 'C1')
    ax.plot(t, y3[:,0], color = 'C2')
    ax.plot(t, y4[:,0], color = 'C3')

    ax.fill_between(t, y1[:,0] - y1[:,1], y1[:,0] + y1[:,1], color='C0', alpha=0.4)
    ax.fill_between(t, y2[:,0] - y2[:,1], y2[:,0] + y2[:,1], color='C1', alpha=0.4)
    ax.fill_between(t, y3[:,0] - y3[:,1], y3[:,0] + y3[:,1], color='C2', alpha=0.4)
    ax.fill_between(t, y4[:,0] - y4[:,1], y4[:,0] + y4[:,1], color='C3', alpha=0.4)

    ax.set_ylabel('Minimum Distance') 
    ax.set_xlabel('Generation')
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=3)
    plt.xticks(np.arange(0, 101, 10))
    plt.title("Minimum Distance to Utopian Point")
    plt.legend(['Control','Random','Chimera','Hypervolume'], 
        loc = 'best', prop={'size': 12})
    plt.savefig('utopia_' + ext + '.jpg', dpi = 400)
# %%
plot_shade(ctrl_mindist,rand_mindist,chim_mindist,hv_mindist,'1')
# %%
# %%
# Task 3: Plotting hypervolumes of Pareto fronts
df = pd.read_csv("all_hvs.csv")
#df = pd.read_csv(pwd + "250k_rndm_zinc_drugs_clean_3.csv")
all_hv = np.empty((10,4))
all_hv[0] = list(df['1'])[:4]
# %%
all_hv[1] = list(df['2'])[:4]
all_hv[2] = list(df['3'])[:4]
all_hv[3] = list(df['4'])[:4]
all_hv[4] = list(df['5'])[:4]
all_hv[5] = list(df['6'])[:4]
all_hv[6] = list(df['7'])[:4]
all_hv[7] = list(df['8'])[:4]
all_hv[8] = list(df['9'])[:4]
all_hv[9] = list(df['10'])[:4]
# %%
df = {}
volumes = np.row_stack((all_hv[:,0],all_hv[:,1],all_hv[:,2],all_hv[:,3]))
types = np.row_stack((["ctrl"]*10,["random"]*10,["chimera"]*10,["hv"]*10))
df["hypervolume"] = volumes.flatten()
df["type"] = types.flatten()
# %%
dframe = pd.DataFrame(data=df)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
sns.violinplot(data=dframe, x="type", y="hypervolume")
plt.ylabel('Hypervolume', fontsize=20)
plt.xlabel('Approach', fontsize=20)
plt.title("Hypervolume of Pareto Front", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(pwd + 'hv_of_pareto_violin.jpg')
# %%
