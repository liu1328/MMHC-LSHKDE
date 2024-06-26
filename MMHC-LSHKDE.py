import networkx as nx
from matplotlib import pyplot as plt
from network import Graph_Matrix
from Directed import hc
from MMPC-LSHKDE import MMPC
import numpy as np

def symmetry(pc):
    """Check the symmetry of the PC to remove false positives"""
    for var in pc:
        for par in pc[var]:
            if var not in pc[par]:
                pc[par].append(var)
    return pc


def MMHC(data, alpha=0.01):
    """
    Parameters
    ----------
    data (np.ndarray): The dataset.
    alpha (float): The significance level for independence tests.

    Returns: Adjacency matrices with directed acyclic graphs
    -------

    """
    _, kvar = np.shape(data)
    DAG = np.zeros((kvar, kvar))
    pc = {}
    for tar in range(kvar):
        pc_mm = MMPC(data, tar, alpha)
        pc[str(tar)] = [str(i) for i in pc_mm]

    pc = symmetry(pc)

    # Use conditional entropy to set the direction
    dag_dict = hc(data, pc)

    for key, value in dag_dict.items():
        x = int(key)
        for i in value['parents']:
            y = int(i)
            DAG[y, x] = 1
            DAG[x, y] = 0
    return DAG

import pandas as pd

if __name__ == '__main__':
    # Read in the data
    data = pd.read_csv('C:\\Users\\kkk\\data1000')
    DAG = MMHC(data)
    print(DAG)
    a = DAG.shape[0]
    nodes1 = np.arange(a)
    nodes = list(map(str,nodes1))
    my_graph = Graph_Matrix(nodes, DAG)
    G = nx.DiGraph()
    for node in my_graph.vertices:
        G.add_node(str(node))
    G.add_weighted_edges_from(my_graph.edges_array)
    pos = nx.shell_layout(G)
    nx.draw(G, with_labels=True,pos=pos)
    # plt.savefig("directed_graph.png")
    plt.show()
