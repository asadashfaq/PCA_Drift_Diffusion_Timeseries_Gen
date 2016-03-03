#! /usr/bin/env python
import os
import aurespf.solvers as au
import numpy as np
from pylab import plt
import multiprocessing as mp
import aurespf.plotting as auplot
import regions.classes as region  ##Added since Node does not exists in auespf anymore.
au = reload(au)
auplot = reload(auplot)

alphas = [0.674268, 0.795557, 0.716455, 0.68291, 0.755273, 0.849023, 0.701074, 0.787354, 0.739453, 0.659473, 0.641748, 0.660791, 0.690674, 0.639551, 0.68862, 0.713086, 0.662549, 0.695068, 0.716016, 0.754102, 0.817676, 0.731543, 0.646582, 0.650977, 0.696533, 0.70708, 0.706787, 0.812915, 0.810791, 0.789551]

## latitudinal ordering of european countries
lat_order = [24,11,10,22,3,9,17,13,4,23,12,0,14,16,26,25,15,6,18,2,8,7,19,29,21,28,20,27,5,1]
## mean load ordering of european countries
size_order =[18,4,7,22,24,20,8,5,2,6,1,15,0,10,14,9,11,12,16,21,17,19,3,26,13,29,27,23,28,25]

# 1 / installed_W gives capacity factor.

installed_W = {'BE': 2.7778, 'FR': 3.125, 'BG': 6.6667, 'DK': 2.4390, 'HR': 5.5556, 'DE': 2.7778, 'HU': 5.2632, 'FI': 3.7037, 'BA': 6.25, 'NL': 2.5, 'PT': 4.5455, 'NO': 3.125, 'LV': 3.4483, 'LT': 3.333, 'LU': 4.0, 'RO': 6.6667, 'PL': 3.125, 'CH': 7.1429, 'GR': 7.1429, 'EE': 3.5714, 'IT': 5.0, 'CZ': 4.7619, 'GB': 2.381, 'IE': 2.1739, 'ES': 5.2632, 'RS': 6.6667, 'SK': 5.8824, 'SI': 6.6667, 'SE': 3.0303, 'AT': 5.2632}

installed_S = {'AT': 5.8824, 'BA': 4.7619, 'BE': 7.1429, 'BG': 4.5455, 'CH': 5.8824, 'CZ': 6.25, 'DE': 6.6667,  'DK': 6.6667, 'EE': 7.6923, 'ES': 4.0, 'FI': 7.6923, 'FR': 5.0, 'GB': 6.6667, 'GR': 4.1667, 'HR': 5.0, 'HU': 5.5556, 'IE': 7.6923, 'IT': 4.3478, 'LT': 7.1429, 'LU': 7.1429, 'LV': 7.6923, 'NL': 6.6667, 'NO': 7.6923, 'PL': 6.6667, 'PT': 4.3478, 'RO': 5.0, 'RS': 5.2632, 'SE': 7.1429, 'SI': 5.5556,'SK': 5.8824}


files = ['AT.npz', 'FI.npz', 'NL.npz', 'BA.npz', 'FR.npz', 'NO.npz', 'BE.npz', 'GB.npz', 'PL.npz', 'BG.npz', 'GR.npz', 'PT.npz', 'CH.npz', 'HR.npz', 'RO.npz', 'CZ.npz', 'HU.npz', 'RS.npz', 'DE.npz', 'IE.npz', 'SE.npz', 'DK.npz', 'IT.npz', 'SI.npz', 'ES.npz', 'LU.npz', 'SK.npz', 'EE.npz', 'LV.npz', 'LT.npz']

def EU_Nodes(load_filename=None, full_load=False):
    return region.Nodes(admat='./settings/eadmat.txt', path='./data/', prefix = "ISET_country_", files=files, load_filename=load_filename, full_load=full_load, alphas=alphas, gammas=np.ones(30))

def EU_Nodes_Gamma_Alpha_input(load_filename=None, g=1.0, a=0.8, full_load=False):
    return region.Nodes(admat='./settings/eadmat.txt', path='./data/', prefix = "ISET_country_", files=files, load_filename=load_filename, full_load=full_load, alphas=a*np.ones(30), gammas=g*np.ones(30))

    
def EU_Nodes_log(load_filename=None, full_load=False,year=0):
    return region.Nodes(admat='./settings/eadmat.txt', path='./data/', prefix = "ISET_country_", files=files, load_filename=load_filename, full_load=full_load, alphas=np.load("./alphas/alpha"+str(year)+".npy"), gammas=np.load("./gammas/gamma"+str(year-2014)+".npy"))
    
def linfo():
    link_info = open("line_info")
    LI = []
    for l in link_info:
        LI.append([l[0:8], l[12:14], int(l[15:-1])])
    return LI


a_first = [0.716016 , 0.674268 , 0.716455 , 0.755273 , 0.739453 , 0.690674 , 0.713086 , 0.817676 , 0.731543 , 0.70708, 0.701074]


def DE_first(load_filename=None, full_load=False):
    N = au.Nodes(admat='./settings/de_first.txt', path='./data/', prefix = "ISET_country_", files=['DE.npz','AT.npz', 'NL.npz', 'FR.npz', 'PL.npz', 'CH.npz', 'CZ.npz', 'SE.npz', 'DK.npz', 'LU.npz', 'BE.npz'], load_filename=load_filename, full_load=full_load, alphas=a_first, gammas=np.ones(11))
    #for n in N:
    #    n.set_alpha(au.optimal_mix_balancing(n.load,n.normwind,n.normsolar)[0])
    return N


