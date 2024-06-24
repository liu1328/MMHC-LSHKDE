
from LSH_KDE import FastGaussianKDE
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

def KDE_xy(x, y):
    xy = np.array([x, y]).T
    hash = FastGaussianKDE(xy, bandwidth=(4/(4*len(x)))**(1/6)-0.1, L=int(len(x)*0.1))
    xy_kde1=[]
    for i in range(len(xy)):
        xy_kde = hash.kde(xy[i, :], bandwidth=(4/(4*len(x)))**(1/6)-0.1)
        xy_kde1.append(xy_kde)
    return xy_kde1

def KDE_xyz(x, y, z):
    xyz = np.array([x, y, z]).T
    hash = FastGaussianKDE(xyz, bandwidth=(4/(5*len(x)))**(1/7)-0.1, L=int(len(x) * 0.1))
    xyz_kde1 = []
    for i in range(len(xyz)):
        xyz_kde = hash.kde(xyz[i, :], bandwidth=(4/(5*len(x)))**(1/7)-0.1)
        xyz_kde1.append(xyz_kde)
    return xyz_kde1

def KDE_xyzz(x, y, z1, z2):
    xyzz = np.array([x, y, z1, z2]).T
    hash = FastGaussianKDE(xyzz, bandwidth=(4/(6*len(x)))**(1/8)-0.1, L=int(len(x) * 0.1))
    xyzz_kde1 = []
    for i in range(len(xyzz)):
        xyzz_kde = hash.kde(xyzz[i, :], bandwidth=(4/(6*len(x)))**(1/8)-0.1)
        xyzz_kde1.append(xyzz_kde)
    return xyzz_kde1

def KDE_xyzzz(x, y, z1, z2, z3):
    xyzzz = np.array([x, y, z1, z2, z3]).T
    hash = FastGaussianKDE(xyzzz, bandwidth=(4/(7*len(x)))**(1/9)-0.1, L=int(len(x) * 0.1))
    xyzzz_kde1 = []
    for i in range(len(xyzzz)):
        xyzzz_kde = hash.kde(xyzzz[i, :], bandwidth=(4/(7*len(x)))**(1/9)-0.1)
        xyzzz_kde1.append(xyzzz_kde)
    return xyzzz_kde1

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
    if len(s) == 2:
        """Calculate conditional mutual information I(x,y|z1,z2)"""
        pxyzz = KDE_xyzz([row[x] for row in data_matrix], [row[y] for row in data_matrix],
                       [row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix])
        pxzz = KDE_xyz([row[x] for row in data_matrix], [row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix])
        pyzz = KDE_xyz([row[y] for row in data_matrix], [row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix])
        pzz = KDE_xy([row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix])
        Ixy_z = np.mean(np.log2(np.divide(np.multiply(pxyzz, pzz), np.multiply(pxzz, pyzz))), dtype="float64")
    if len(s) == 3:
        """Calculate conditional mutual information I(x,y|z1,z2,z3)"""
        pxyzzz = KDE_xyzzz([row[x] for row in data_matrix], [row[y] for row in data_matrix],
                       [row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix], [row[s[2]] for row in data_matrix])
        pxzzz = KDE_xyzz([row[x] for row in data_matrix], [row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix], [row[s[2]] for row in data_matrix])
        pyzzz = KDE_xyzz([row[y] for row in data_matrix], [row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix], [row[s[2]] for row in data_matrix])
        pzzz = KDE_xyz([row[s[0]] for row in data_matrix], [row[s[1]] for row in data_matrix], [row[s[2]] for row in data_matrix])
        Ixy_z = np.mean(np.log2(np.divide(np.multiply(pxyzzz, pzzz), np.multiply(pxzzz, pyzzz))), dtype="float64")

    if Ixy_z < alpha:
        dep = 0
    else:
        dep = Ixy_z
    return Ixy_z, dep





