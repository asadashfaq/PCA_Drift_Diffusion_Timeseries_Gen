__author__ = 'Raunbak'
import numpy as np
import regions.classes as region
import tools as common_tools
files = ['AT.npz', 'FI.npz', 'NL.npz', 'BA.npz', 'FR.npz', 'NO.npz', 'BE.npz', 'GB.npz', 'PL.npz', 'BG.npz', 'GR.npz', 'PT.npz', 'CH.npz', 'HR.npz', 'RO.npz', 'CZ.npz', 'HU.npz', 'RS.npz', 'DE.npz', 'IE.npz', 'SE.npz', 'DK.npz', 'IT.npz', 'SI.npz', 'ES.npz', 'LU.npz', 'SK.npz', 'EE.npz', 'LV.npz', 'LT.npz']


def get_eu_mismatch_balancing_injection_meanload(filename='results/Europe_gHO1.0_aHO0.8_Linear.npz'):

    data_load = np.load(filename)
    balancing = data_load['curtailment'] - data_load['balancing']
    mismatch = data_load['mismatch'] - data_load['storage_discharge']
    pure_balancing = data_load['balancing']

    # Can't get to the injection without using functions on nodes.
    #N = region.Nodes(admat='./settings/eadmat.txt', path='./data/', prefix="ISET_country_", files=files, load_filename=filename[8:], full_load=False)

    Kmatrix = AtoK_simple(pathadmat='./settings/eadmat.txt')
    # From indice matrix to flow.
    #K = common_tools.AtoKh_old(N)[0]

    flowfile = filename[:-4]+'_Flow.npy'
    F = np.load(flowfile)
    injection = np.dot(Kmatrix, F)

    # Calculate meanload
    meanload = np.sum(np.mean(data_load['load'], axis=1))

    return mismatch, balancing, injection, pure_balancing, meanload


def get_country_mean_load(filename='results/Europe_gHO1.0_aHO0.8_Linear.npz',country=0):
    data_load = np.load(filename)
    meanload = np.mean(data_load['load'][country,:])
    return meanload

def AtoK_simple(pathadmat=None):
    Ad = np.genfromtxt(pathadmat,dtype='d')
    L = 0
    for j in range(len(Ad)):
        for i in range(len(Ad)):
            if i>j:
                if Ad[i,j] > 0:
                    L+=1

    K = np.zeros((len(Ad),L))
    L = 0

    for j in range(len(Ad)):
        for i in range(len(Ad)):
            if i>j:
                if Ad[i,j] > 0:
                    K[j,L]=1
                    K[i,L]=-1
                    L+=1

    return K