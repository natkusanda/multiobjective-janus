
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
from cairosvg import svg2png
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
'''
Task 1: getting hypervolumes, retain 10% percentile worst molecules
'''
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
with open("zinc_250k_sliced.txt", "w") as f:
    f.writelines(["{}".format(x) for x in hv_keep_smiles])
# %%
'''
Task 2: Getting minimum distances to utopian point
'''
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
# %% Plotting minimum distance
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

    ax.set_ylabel('Minimum Distance', fontsize=25) 
    ax.set_xlabel('Generation', fontsize=25)
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=3)
    plt.xticks(np.arange(0, 101, 10))
    plt.title("Minimum Distance to Utopian Point")
    plt.legend(['Control','Random','Chimera','Hypervolume'], 
        loc = 'best', prop={'size': 25})
    plt.savefig('utopia_' + ext + '.jpg', dpi = 400)
# %%
plot_shade(ctrl_mindist,rand_mindist,chim_mindist,hv_mindist,'1')
# %%
# %%
'''
Task 3: Plotting hypervolumes of Pareto fronts
'''
df = pd.read_csv("all_hvs.csv")
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
# %% Violin plot
dframe = pd.DataFrame(data=df)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
sns.violinplot(data=dframe, x="type", y="hypervolume")
plt.axhline(y=12.40,color='black', label = 'ZINC_red average',linestyle='--')
ax.annotate('ZINC_red',xy=(2.4,12.45),fontsize=20)
plt.ylabel('Hypervolume', fontsize=20)
plt.xlabel('Approach', fontsize=20)
plt.xticks(rotation = 15)
plt.title("Hypervolume of Pareto Front", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('hv_of_pareto_violin.png')
# %%
'''
Task 4: Plotting closest molecules to utopian point
'''
def top_min_all(optim_type):
    pwd = "/RESULTS/"
    min_distances = []

    min_qed = []
    min_logp = []
    min_dist = []
    min_smi = []

    for i in tqdm(range(1,11)):
        
        if i == 1:
            df = pd.read_csv(pwd + "smiles_collector_" + optim_type + ".csv")
        else:
            df = pd.read_csv(pwd + "smiles_collector_" + optim_type + str(i) + ".csv")


        smiles = list(df['smi'])
        logp = list(df['logp'])
        qed = list(df['qed'])

        data = np.column_stack((qed,logp))
        utopia = np.array([0.6,10])
        dist_1 = (utopia[0] - data[:,0]) / utopia[0]
        dist_2 = (utopia[1] - data[:,1]) / utopia[1]

        dist_1 = np.array([dist_1[i] if dist_1[i]>0 else 0 for i in range(len(dist_1))])
        dist_2 = np.array([dist_2[i] if dist_2[i]>0 else 0 for i in range(len(dist_2))])

        all_dist = np.sqrt(dist_1 ** 2 + dist_2 ** 2)

        sort_by_dist_idx = np.argsort(all_dist)

        sort_qed = np.array(qed)[sort_by_dist_idx]
        sort_logp = np.array(logp)[sort_by_dist_idx]
        sort_dist = np.array(all_dist)[sort_by_dist_idx]
        sort_smi = np.array(smiles)[sort_by_dist_idx]

        min_qed.extend(list(sort_qed[:10]))
        min_logp.extend(list(sort_logp[:10]))
        min_dist.extend(list(sort_dist[:10]))
        min_smi.extend(list(sort_smi[:10]))
    
    final_sort_idx = np.argsort(min_dist)
    top_qed = np.array(min_qed)[final_sort_idx]
    top_logp = np.array(min_logp)[final_sort_idx]
    top_dist = np.array(min_dist)[final_sort_idx]
    top_smi = np.array(min_smi)[final_sort_idx]

    return top_qed[:30], top_logp[:30], top_dist[:30], top_smi[:30]
# %%
chim_three = np.empty((3,30))
chim_three[0],chim_three[1],chim_three[2],chim_smiles = top_min_all('chimera')
# %%
ctrl_three = np.empty((3,30))
ctrl_three[0],ctrl_three[1],ctrl_three[2],ctrl_smiles = top_min_all('ctrl')
# %%
random_three = np.empty((3,30))
random_three[0],random_three[1],random_three[2],rand_smiles = top_min_all('random')
# %%
hv_three = np.empty((3,30))
hv_three[0],hv_three[1],hv_three[2],hv_smiles = top_min_all('hv')
# %%
def top_zinc_pareto_points():
    min_distances = []

    min_qed = []
    min_logp = []
    min_dist = []
    min_smi = []

    df = pd.read_csv("pareto_tagged_logp_qed.csv")


    smiles = list(df['smiles'])
    logp = list(df['logP'])
    qed = list(df['qed'])
    pareto = list(df['pareto'])

    qed_pareto = [qed[i] for i in range(len(smiles)) if pareto[i] == 1]
    logp_pareto = [logp[i] for i in range(len(smiles)) if pareto[i] == 1]
    smiles_pareto = [smiles[i] for i in range(len(smiles)) if pareto[i] == 1]

    data = np.column_stack((qed_pareto,logp_pareto))
    utopia = np.array([0.6,10])
    dist_1 = (utopia[0] - data[:,0]) / utopia[0]
    dist_2 = (utopia[1] - data[:,1]) / utopia[1]

    dist_1 = np.array([dist_1[i] if dist_1[i]>0 else 0 for i in range(len(dist_1))])
    dist_2 = np.array([dist_2[i] if dist_2[i]>0 else 0 for i in range(len(dist_2))])

    all_dist = np.sqrt(dist_1 ** 2 + dist_2 ** 2)

    sort_by_dist_idx = np.argsort(all_dist)
    
    sort_qed = np.array(qed_pareto)[sort_by_dist_idx]
    sort_logp = np.array(logp_pareto)[sort_by_dist_idx]
    sort_dist = np.array(all_dist)[sort_by_dist_idx]
    sort_smi = np.array(smiles_pareto)[sort_by_dist_idx]

    min_qed.extend(list(sort_qed[:30]))
    min_logp.extend(list(sort_logp[:30]))
    min_dist.extend(list(sort_dist[:30]))
    min_smi.extend(list(sort_smi[:30]))
    

    return min_qed,min_logp,min_dist,min_smi
# %%
zinc_pareto = np.empty((3,30))
zinc_pareto[0],zinc_pareto[1],zinc_pareto[2],zinc_smi = top_zinc_pareto_points()
# %%
df = {}
df['qed'] = np.concatenate((ctrl_three[0],random_three[0],chim_three[0],hv_three[0]))
df['logp'] = np.concatenate((ctrl_three[1],random_three[1],chim_three[1],hv_three[1]))
df['type'] = np.concatenate((['WeightedSum']*30,['Random']*30,['Chimera']*30,['Hypervolume']*30))
utop_df = {}
utop_df['qed'] = [0.6]
utop_df['logp'] = [10]
utop_df['type'] = ['Utopian point']
pareto_df = {}
pareto_df['qed'] = zinc_pareto[0]
pareto_df['logp'] = zinc_pareto[1]
pareto_df['type'] = ['Zinc'] * 30
# %%
def good_plot(ml_df, utop_df, pareto_df, arg1, arg2, ext, xlims, ylims):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.axhline(0, linestyle="-", color='black')
    ax.axvline(0, linestyle="-", color='black')
    plt.title("Plot of molecules closest to utopian point")

    markers = {"Zinc": 's',"Chimera": 'o',"WeightedSum": 'o',
        "Random": 'o',"Hypervolume": 'o',"Utopian point": 'X'}
    sns.scatterplot(data=ml_df, x=arg1, y=arg2, hue ='type', 
           legend ='full',style='type',markers=markers,s=60)
    sns.scatterplot(data=utop_df, x=arg1, y=arg2, hue='type',palette='gist_gray_r',
           legend ='full',style='type',markers=markers,s=500)
    sns.lineplot(data=pareto_df, x=arg1, y=arg2, hue ='type',palette='Purples', sizes=(50,50),
           legend ='full',style='type',markers=markers,markersize=10)
    plt.legend(loc = 'best', prop={'size': 20})
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    ax.set_ylabel('QED', fontsize=20) 
    ax.set_xlabel('logP', fontsize=20)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.savefig("utopia" + arg1 + '_' + arg2 + '_' + ext + '.png',dpi = 800)
# %%
ml_df = pd.DataFrame(df)
u_df = pd.DataFrame(utop_df)
p_df = pd.DataFrame(pareto_df)
good_plot(ml_df, u_df, p_df, 'logp','qed','close_plot',[4,10.5],[0.3,0.9])

# %%
'''
Task 5: Visualising generated molecules
'''
# %%
def visualise_grid(optim_smiles,optim_data,optim_type):
    subms = [Chem.MolFromSmiles(x) for x in optim_smiles]
    legends = []
    for i in range(len(optim_smiles)):
        
        qed_str = "QED: {:.2f}".format(optim_data[0][i])
        logp_str = "logP: {:.2f}".format(optim_data[1][i])

        #legend = chim_smiles[i] + "\n" * 5 + qed_str + "\n" * 5 + logp_str
        legend = qed_str + "\n" * 5 + logp_str
        legends.append(legend)
    
    d2d = Chem.Draw.MolDraw2DSVG(600 * 5, 600 * 2, 600, 600)
    d2d.drawOptions().addStereoAnnotation=True
    d2d.drawOptions().legendFontSize = 55
    d2d.drawOptions().legendFraction = 0.2
    d2d.DrawMolecules(subms[:10],legends=legends[:10])
    d2d.FinishDrawing()
    svg_string = d2d.GetDrawingText()
    svg2png(bytestring=svg_string,write_to=optim_type+'_pop.png')
# %%
visualise_grid(chim_smiles,chim_three,'chimera')
visualise_grid(ctrl_smiles,ctrl_three,'ctrl')
visualise_grid(rand_smiles,random_three,'random')
visualise_grid(hv_smiles,hv_three,'hv')