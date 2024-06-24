import numpy as np
import pandas as pd

from condition_independence_test import cond_indep_test
from subsets import subsets

def getMinDep(data, target, x, CPC, alpha):
    """this function is to chose min dep(association) about Target,x|(subsets of CPC)"""
    dep_min = float("inf")
    max_k = 3
    # It is rare for a node to have more than three Perents or children (one of them).
    if len(CPC) > max_k:
        k_length = max_k
    else:
        k_length = len(CPC)
    for i in range(k_length+1):
        SS = subsets(CPC, i)
        for S in SS:
            pval, dep = cond_indep_test(data, target, x, S)
            if pval < alpha:
                return 0
            if dep_min > dep:
                dep_min = dep
    return dep_min


def MMPC(data, target, alpha):
    """
        Find the parent and child nodes of each node

        Args:
            data (np.ndarray): The dataset.
            target (int): The index of the target variable.
            alpha (float): The significance level for independence tests.

        Returns:
            dict: A dictionary containing the MMPC parameters.
        """
    number, kVar = np.shape(data)
    CPC = []
    deoZeroSet = []

    """phaseI :forward"""

    while True:
        M_variables = [i for i in range(kVar) if i != target and i not in CPC and i not in deoZeroSet]
        vari_all_dep_max = -float("inf")
        vari_chose = 0

        for x in M_variables:
            # Use the getMinDep function to select the minimum dep for x
            x_dep_min= getMinDep(data, target, x, CPC, alpha)

            # If x has a dep of 0, it can never be included in CPC, and it can be added to deoZeroSet
            if x_dep_min == 0:
                deoZeroSet.append(x)

            # Select the largest variable
            elif x_dep_min > vari_all_dep_max:
                vari_chose = x
                vari_all_dep_max = x_dep_min

        # Select the largest variable with a score greater than =0 and add it to the cpc
        if vari_all_dep_max >= 0:
            CPC.append(vari_chose)
        else:
            break

    """phaseII :Backward"""

    CPC_temp = CPC.copy()
    max_k = 3
    for a in CPC_temp:
        C_subsets = [i for i in CPC if i != a]
        if len(C_subsets) > max_k:
            C_length = max_k
        else:
            C_length = len(C_subsets)

        breakFlag = False
        for length in range(C_length + 1):
            if breakFlag:
                break
            SS = subsets(C_subsets, length)
            for S in SS:
                pval, dep = cond_indep_test(data,target, a, S)
                # If pval is less than alpha, it is removed from the CPC
                if pval < alpha:
                    CPC.remove(a)
                    breakFlag = True
                    break
    return list(set(CPC))

import networkx as nx
from matplotlib import pyplot as plt
from network import Graph_Matrix

# if __name__ == '__main__':
#     # Read in the data
#     data = pd.read_csv('C:\\Users\\kkk\\Downloads\\pyCausalFS\\pyCausalFS1\\pyCausalFS\\GSL\\MMHC\\Besian\\data1000')
#     _, kvar = np.shape(data)
#     DAG = np.zeros((kvar, kvar))
#     adjacency_matrix = np.zeros((kvar, kvar))
#     alpha = 0.01
#     pc = {}
#     for tar in range(kvar):
#         print("tar", tar)
#         pc_mm = MMPC(data, tar, alpha)
#         # 把MMPC的结果转换成字符串存入PC中
#         pc[str(tar)] = [str(i) for i in pc_mm]
#     print(pc)
