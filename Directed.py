import numpy as np
import copy
from LSH_KDE import FastGaussianKDE

def score_diff(gra1, gra2, data):
    """Keep the data you need"""
    remain = []
    for tar in range(data.shape[1]):
        if set(gra1[str(tar)]) != set(gra2[str(tar)]):
            remain.append(str(tar))
            remain.extend(gra1[str(tar)])
            remain.extend(gra2[str(tar)])
    remain = list(set(remain))

    sco_diff = C_Entropy(gra1, gra2, data[remain])
    return sco_diff

np.random.seed(101)

def H(data_matrix, tar):
    x1 = data_matrix[str(tar)]
    x = np.array([x1]).T
    hash = FastGaussianKDE(x, bandwidth=(4/(3*len(x)))**(1/5)-0.1, L=int(len(x) * 0.05))
    x_kde1 = []
    for i in range(len(x)):
        x_kde = hash.kde(x[i, :], bandwidth=(4/(3*len(x)))**(1/5)-0.1)
        x_kde1.append(x_kde)
    return x_kde1

def Hxy(data_matrix, tar, p):
    x = data_matrix[str(tar)]
    y = data_matrix[str(p)]
    xy = np.array([x, y]).T
    hash = FastGaussianKDE(xy, bandwidth=(4/(4*len(x)))**(1/6)-0.1, L=int(len(x)*0.05))
    xy_kde1=[]
    for i in range(len(xy)):
        xy_kde = hash.kde(xy[i, :], bandwidth=(4/(4*len(x)))**(1/6)-0.1)
        xy_kde1.append(xy_kde)
    return xy_kde1

def Hxyz(data_matrix, tar, p1, p2):
    x = data_matrix[str(tar)]
    y = data_matrix[str(p1)]
    z = data_matrix[str(p2)]
    xyz = np.array([x, y, z]).T
    hash = FastGaussianKDE(xyz, bandwidth=(4/(5*len(x)))**(1/7)-0.1, L=int(len(x) * 0.05))
    xyz_kde1 = []
    for i in range(len(xyz)):
        xyz_kde = hash.kde(xyz[i, :], bandwidth=(4/(5*len(x)))**(1/7)-0.1)
        xyz_kde1.append(xyz_kde)
    return xyz_kde1

def Hxyzz(data_matrix, tar, p1, p2, p3):
    x = data_matrix[str(tar)]
    y = data_matrix[str(p1)]
    z = data_matrix[str(p2)]
    z1 = data_matrix[str(p3)]
    xyzz = np.array([x, y, z, z1]).T
    hash = FastGaussianKDE(xyzz, bandwidth=(4/(6*len(x)))**(1/8)-0.1, L=int(len(x) * 0.05))
    xyz_kde1 = []
    for i in range(len(xyzz)):
        xyz_kde = hash.kde(xyzz[i, :], bandwidth=(4/(6*len(x)))**(1/8)-0.1)
        xyz_kde1.append(xyz_kde)
    return xyz_kde1

def C_Entropy(gra1, gra2, data):
    "Start calculating conditional entropy"
    score_diff=0
    for i in data:
        score_con=0
        score_con1=0
        if set(gra1[str(i)]) != set(gra2[str(i)]):
            if len(gra1[str(i)]) == 0:
                "There is no parent node, and the scoring function is to calculate a single entropy"
                if len(gra2[str(i)]) == 1:
                    score1 = Hxy(data, i, gra2[str(i)][0])
                    score2 = H(data, gra2[str(i)][0])
                    score_con1 = (-1) * np.mean(np.log(score1)) - (-1) * np.mean(np.log(score2))
                    score = H(data, i)
                    score_con = (-1) * np.mean(np.log(score))

            if len(gra2[str(i)]) == 2:
                if len(gra1[str(i)]) == 1:
                    score1 = Hxy(data, i, gra1[str(i)][0])
                    score2 = H(data, gra1[str(i)][0])
                    score_con = (-1) * np.mean(np.log(score1)) - ((-1) * np.mean(np.log(score2)))

                    score3 = Hxyz(data, i, gra2[str(i)][0], gra2[str(i)][1])
                    score4 = Hxy(data, gra2[str(i)][0], gra2[str(i)][1])
                    score_con1 = (-1) * np.mean(np.log(score3)) - (-1) * np.mean(np.log(score4))

            if len(gra2[str(i)]) == 3:
                if len(gra1[str(i)]) == 2:
                    score3 = Hxyz(data, i, gra1[str(i)][0], gra1[str(i)][1])
                    score4 = Hxy(data, gra1[str(i)][0], gra1[str(i)][1])
                    score_con = (-1) * np.mean(np.log(score3)) - (-1) * np.mean(np.log(score4))

                    score5 = Hxyzz(data, i, gra2[str(i)][0], gra2[str(i)][1], gra2[str(i)][2])
                    score6 = Hxyz(data, gra2[str(i)][0], gra2[str(i)][1], gra2[str(i)][2])
                    score_con1 = (-1) * np.mean(np.log(score5)) - (-1) * np.mean(np.log(score6))

            score_diff = score_con - score_con1

    return score_diff

def check_cycle(tar, pc_var, gra):
    "Check for rings"
    underchecked = [x for x in gra[str(tar)] if x != pc_var]
    checked = []
    cyc_flag = False
    while underchecked:
        if cyc_flag:
            break
        underchecked_copy = list(underchecked)
        for vk in underchecked_copy:
            if gra[vk]:
                if pc_var in gra[vk]:
                    cyc_flag = True
                    break
                else:
                    for key in gra[vk]:
                        if key not in checked + underchecked:
                            underchecked.append(key)
            underchecked.remove(vk)
            checked.append(vk)
    return cyc_flag

def hc(data, pc):
    """
    Conditional entropy is used for scoring
    Parameters
    ----------
    data: The dataset.
    pc: The parent and child nodes of each node.

    Returns: Directed acyclic graph
    -------

    """
    gra = {}
    gra_temp = {}
    for node in range(data.shape[1]):
        gra[str(node)] = []
        gra_temp[str(node)] = []
    diff = 1

    # Try to find the best diagram until nothing changes
    while diff > 1e-10:
        diff = 0
        edge_candidate = []
        gra_temp = copy.deepcopy(gra)

        for tar in range(data.shape[1]):
            for pc_var in pc[str(tar)]:
                cyc_flag = check_cycle(str(tar), pc_var, gra)
                if not cyc_flag:
                    gra_temp[str(tar)].append(pc_var)
                    score_diff_temp = score_diff(gra, gra_temp, data)
                    if (score_diff_temp - diff > -1e-10):
                        diff = score_diff_temp
                        edge_candidate = [str(tar), pc_var, 'a']

                    gra_temp[str(tar)].remove(pc_var)

        if edge_candidate:
            if edge_candidate[2] == 'a':
                gra[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[1]].remove(edge_candidate[0])


    dag = {}
    for var in gra:
        dag[var] = {}
        dag[var]['parents'] = gra[var]
    return dag






