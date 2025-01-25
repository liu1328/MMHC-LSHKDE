import logging
import numpy as np
import warnings
warnings.filterwarnings('ignore')
_logger = logging.getLogger(__name__)

#
np.random.seed(1)

def KDE_x(x):
    x = np.array([x]).T
    hash = FastGaussianKDE(x, bandwidth=(4/(3*len(x)))**(1/5)-0.1, L=int(len(x)*0.1))
    x_kde1 = []
    for i in range(len(x)):
        x_kde = hash.kde(x[i, :], bandwidth=(4/(3*len(x)))**(1/5)-0.1)
        x_kde1.append(x_kde)
    return x_kde1

def g_MI(data_matrixx, x, y, s, alpha):
    data_matrix = np.array(data_matrixx)
    Ixy_z = 0
    if len(s) == 0:
        """Computational Mutual Information"""
        pxy = KDE_xy([row[x] for row in data_matrix], [row[y] for row in data_matrix])
        px = KDE_x([row[x] for row in data_matrix])
        py = KDE_x([row[y] for row in data_matrix])
        px_py = np.multiply(px, py)
        Ixy_z = np.mean(np.log2(np.divide(pxy, px_py)), dtype="float64")
    if len(s) == 1:
        """Calculate conditional mutual information I(x,y|z1)"""
        pxyz = KDE_xyz([row[x] for row in data_matrix], [row[y] for row in data_matrix],
                           [row[s[0]] for row in data_matrix])
        pxz = KDE_xy([row[x] for row in data_matrix], [row[s[0]] for row in data_matrix])
        pyz = KDE_xy([row[y] for row in data_matrix], [row[s[0]] for row in data_matrix])
        pz = KDE_x([row[s[0]] for row in data_matrix])
        Ixy_z = np.mean(np.log2(np.divide(np.multiply(pxyz, pz), np.multiply(pxz, pyz))), dtype="float64")

    if Ixy_z < alpha:
        dep = 0
    else:
        dep = Ixy_z
    return Ixy_z, dep





